/*******************************************************************************
 * Algorithm: ERASOR2-Fins (Denoised & Intensity Preserved)
 * Features: 3D Voxel Clustering + Noise Filter + Full Intensity Support
 ******************************************************************************/

#include <fins/node.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <omp.h>

#if __has_include(<tf2_eigen/tf2_eigen.hpp>)
  #include <tf2_eigen/tf2_eigen.hpp>
#else
  #include <tf2_eigen/tf2_eigen.h>
#endif

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <bitset>
#include <unordered_map>
#include <unordered_set>
#include <queue>

namespace erasor2_final {

struct __attribute__((packed)) PointXYZI {
    float x; float y; float z; float intensity;
};

struct PointInternal {
    Eigen::Vector3d pos_w; 
    float intensity;
};

using VDescriptor = std::bitset<64>;

struct Bin {
    float max_z = -99.0f;
    int count = 0;
    VDescriptor v_desc;
    bool is_dynamic_bin = false;

    void update(float z, float ground_z, float v_res) {
        if (z > max_z) max_z = z;
        count++;
        int v_idx = static_cast<int>((z - ground_z) / v_res);
        if (v_idx >= 0 && v_idx < 64) v_desc.set(v_idx);
    }
    void reset() {
        max_z = -99.0f; count = 0; v_desc.reset(); is_dynamic_bin = false;
    }
};

class ErasorEngine {
public:
    struct Params {
        int num_sectors = 360;
        int num_rings = 60;
        float v_res = 0.1f;
        float bsr_threshold = 0.6f;
        float h_threshold = 0.15f;
        float cluster_res = 0.4f;
        float max_range = 40.0f;
        float max_dynamic_h = 2.5f;
        int min_cluster_size = 50; 
    };

    ErasorEngine(const Params& p) : p_(p) {
        bins_map_.resize(p_.num_sectors * p_.num_rings);
        bins_curr_.resize(p_.num_sectors * p_.num_rings);
    }

    inline int get_idx(double x, double y) const {
        double r = std::sqrt(x*x + y*y);
        if (r > p_.max_range || r < 0.4) return -1;
        double a = std::atan2(y, x) + M_PI;
        int s_idx = static_cast<int>(a / (2.0 * M_PI) * (p_.num_sectors - 1));
        int r_idx = static_cast<int>(r / p_.max_range * (p_.num_rings - 1));
        return r_idx * p_.num_sectors + s_idx;
    }

    void reset_curr() {
        #pragma omp parallel for
        for (size_t i = 0; i < bins_curr_.size(); ++i) bins_curr_[i].reset();
    }

    void fill_curr(const std::vector<PointInternal>& pts, float ground_z) {
        for (const auto& p : pts) {
            int idx = get_idx(p.pos_w.x(), p.pos_w.y());
            if (idx != -1 && p.pos_w.z() < ground_z + p_.max_dynamic_h) {
                bins_curr_[idx].update((float)p.pos_w.z(), ground_z, p_.v_res);
            }
        }
    }

    void compute_dynamic_bins() {
        #pragma omp parallel for
        for (size_t i = 0; i < bins_curr_.size(); ++i) {
            auto& c = bins_curr_[i]; auto& m = bins_map_[i];
            if (c.count < 2 || m.count == 0) continue;
            if (c.max_z > m.max_z + p_.h_threshold) {
                VDescriptor robust_m = m.v_desc | (m.v_desc << 1) | (m.v_desc >> 1);
                float bsr = (float)(c.v_desc & robust_m).count() / (float)c.v_desc.count();
                if (bsr < p_.bsr_threshold) c.is_dynamic_bin = true;
            }
        }
    }

    void apply_clustering(const std::vector<PointInternal>& pts, std::vector<int8_t>& is_dyn, float ground_z) {
        std::unordered_map<int64_t, std::vector<int>> voxel_map;
        double inv_res = 1.0 / (double)p_.cluster_res;

        for(size_t i=0; i<pts.size(); ++i) {
            if (pts[i].pos_w.z() > ground_z + p_.max_dynamic_h) continue;
            int64_t ix = (int64_t)std::floor(pts[i].pos_w.x() * inv_res);
            int64_t iy = (int64_t)std::floor(pts[i].pos_w.y() * inv_res);
            int64_t iz = (int64_t)std::floor(pts[i].pos_w.z() * inv_res);
            int64_t k = (ix + 1000) << 40 | (iy + 1000) << 20 | (iz + 1000);
            voxel_map[k].push_back(i);
        }

        std::unordered_set<int64_t> dynamic_voxels;
        for(auto& kv : voxel_map) {
            int dyn_votes = 0;
            for(int pt_idx : kv.second) {
                int bin_idx = get_idx(pts[pt_idx].pos_w.x(), pts[pt_idx].pos_w.y());
                if (bin_idx != -1 && bins_curr_[bin_idx].is_dynamic_bin) dyn_votes++;
            }
            if (dyn_votes > 0 && (float)dyn_votes / (float)kv.second.size() > 0.3f) {
                dynamic_voxels.insert(kv.first);
            }
        }

        std::unordered_set<int64_t> visited;
        for (int64_t v_key : dynamic_voxels) {
            if (visited.count(v_key)) continue;

            std::vector<int64_t> cluster_voxels;
            std::queue<int64_t> q;
            q.push(v_key);
            visited.insert(v_key);

            while (!q.empty()) {
                int64_t curr = q.front(); q.pop();
                cluster_voxels.push_back(curr);
                int64_t cx = (curr >> 40) - 1000;
                int64_t cy = ((curr >> 20) & 0xFFFFF) - 1000;
                int64_t cz = (curr & 0xFFFFF) - 1000;
                for (int dx=-1; dx<=1; ++dx) {
                    for (int dy=-1; dy<=1; ++dy) {
                        for (int dz=-1; dz<=1; ++dz) {
                            if (dx==0 && dy==0 && dz==0) continue;
                            int64_t nk = (cx+dx+1000) << 40 | (cy+dy+1000) << 20 | (cz+dz+1000);
                            if (dynamic_voxels.count(nk) && !visited.count(nk)) {
                                visited.insert(nk); q.push(nk);
                            }
                        }
                    }
                }
            }

            size_t cluster_point_count = 0;
            for (int64_t vk : cluster_voxels) cluster_point_count += voxel_map[vk].size();

            if (cluster_point_count >= (size_t)p_.min_cluster_size) {
                for (int64_t vk : cluster_voxels) {
                    for (int pt_idx : voxel_map[vk]) is_dyn[pt_idx] = 1;
                }
            }
        }
    }

    void update_map() {
        #pragma omp parallel for
        for (size_t i = 0; i < bins_map_.size(); ++i) {
            auto& c = bins_curr_[i];
            if (c.count > 0 && !c.is_dynamic_bin) {
                if (bins_map_[i].count == 0) bins_map_[i] = c;
                else {
                    bins_map_[i].v_desc |= c.v_desc;
                    bins_map_[i].max_z = std::max(bins_map_[i].max_z, c.max_z);
                    bins_map_[i].count = 1;
                }
            }
        }
    }

private:
    Params p_;
    std::vector<Bin> bins_map_;
    std::vector<Bin> bins_curr_;
};

} // namespace erasor2_final

class ErasorDetectorNode : public fins::Node {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void define() override {
        set_name("ErasorDetector");
        register_input<sensor_msgs::msg::PointCloud2>("lidar_in", &ErasorDetectorNode::on_lidar);
        register_input<geometry_msgs::msg::TransformStamped>("$T_{odom}^{base}$", &ErasorDetectorNode::on_odom);
        register_output<sensor_msgs::msg::PointCloud2>("dynamic_cloud");
        register_output<sensor_msgs::msg::PointCloud2>("static_cloud");
    }

    void initialize() override {
        fins::ParamLoader cfg("ErasorDetector");
        erasor2_final::ErasorEngine::Params p;
        p.h_threshold = cfg.get("height_threshold", 0.15f);
        p.bsr_threshold = cfg.get("bsr_threshold", 0.6f);
        p.cluster_res = cfg.get("cluster_res", 0.4f);
        p.max_dynamic_h = cfg.get("max_dynamic_height", 2.5f);
        p.min_cluster_size = cfg.get("min_cluster_size", 50);
        ground_z_ = cfg.get("ground_z", -0.05f);
        engine_ = std::make_unique<erasor2_final::ErasorEngine>(p);
    }

private:
    void on_odom(const fins::Msg<geometry_msgs::msg::TransformStamped>& m) {
        T_w_b_ = tf2::transformToEigen(*m); received_odom_ = true;
    }

    void on_lidar(const fins::Msg<sensor_msgs::msg::PointCloud2>& msg) {
        if (!received_odom_) return;
        
        // 鲁棒的字段解析
        int x_off = -1, y_off = -1, z_off = -1, i_off = -1;
        for (const auto& f : msg->fields) {
            if (f.name == "x") x_off = f.offset;
            else if (f.name == "y") y_off = f.offset;
            else if (f.name == "z") z_off = f.offset;
            else if (f.name == "intensity") i_off = f.offset;
        }
        if (x_off == -1 || y_off == -1 || z_off == -1) return;

        size_t count = msg->width * msg->height;
        std::vector<erasor2_final::PointInternal> pts_int(count);
        engine_->reset_curr();

        #pragma omp parallel for schedule(dynamic, 512)
        for (size_t i = 0; i < count; ++i) {
            const uint8_t* d = &msg->data[i * msg->point_step];
            float lx = *reinterpret_cast<const float*>(d + x_off);
            float ly = *reinterpret_cast<const float*>(d + y_off);
            float lz = *reinterpret_cast<const float*>(d + z_off);
            float intensity = (i_off != -1) ? *reinterpret_cast<const float*>(d + i_off) : 0.0f;
            
            pts_int[i].pos_w = T_w_b_ * Eigen::Vector3d(lx, ly, lz);
            pts_int[i].intensity = intensity;
        }

        engine_->fill_curr(pts_int, ground_z_);
        engine_->compute_dynamic_bins();

        std::vector<int8_t> is_dyn(count, 0);
        engine_->apply_clustering(pts_int, is_dyn, ground_z_);

        std::vector<erasor2_final::PointXYZI> d_vec, s_vec;
        d_vec.reserve(count / 4); s_vec.reserve(count);

        for (size_t i = 0; i < count; ++i) {
            const uint8_t* d = &msg->data[i * msg->point_step];
            erasor2_final::PointXYZI p;
            p.x = *reinterpret_cast<const float*>(d + x_off);
            p.y = *reinterpret_cast<const float*>(d + y_off);
            p.z = *reinterpret_cast<const float*>(d + z_off);
            p.intensity = pts_int[i].intensity; // 关键：保留从原始 buffer 读取的强度

            if (is_dyn[i] == 1) d_vec.push_back(p); else s_vec.push_back(p);
        }

        engine_->update_map();
        publish(d_vec, "dynamic_cloud", msg->header, msg.acq_time);
        publish(s_vec, "static_cloud", msg->header, msg.acq_time);
    }

    void publish(const std::vector<erasor2_final::PointXYZI>& pts, const std::string& chan, 
                 const std_msgs::msg::Header& header, const fins::AcqTime& acq_time) {
        if (pts.empty()) return;
        sensor_msgs::msg::PointCloud2 m;
        m.header = header;
        m.height = 1; m.width = pts.size();
        m.point_step = 16;
        m.row_step = 16 * m.width;
        m.is_dense = true;
        std::vector<std::string> fn = {"x", "y", "z", "intensity"};
        for (size_t i = 0; i < fn.size(); ++i) {
            sensor_msgs::msg::PointField f;
            f.name = fn[i]; f.offset = i * 4;
            f.datatype = sensor_msgs::msg::PointField::FLOAT32; f.count = 1;
            m.fields.push_back(f);
        }
        m.data.resize(pts.size() * 16);
        std::memcpy(m.data.data(), pts.data(), m.data.size());
        send(chan, m, acq_time);
    }

    Eigen::Isometry3d T_w_b_ = Eigen::Isometry3d::Identity();
    bool received_odom_ = false;
    float ground_z_ = 0.0f;
    std::unique_ptr<erasor2_final::ErasorEngine> engine_;
};

EXPORT_NODE(ErasorDetectorNode)
DEFINE_PLUGIN_ENTRY(fins::STATELESS)