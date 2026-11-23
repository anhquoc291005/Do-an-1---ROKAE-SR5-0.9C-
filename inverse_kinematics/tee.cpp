#include <rclcpp/rclcpp.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <geometric_shapes/shape_operations.h>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <filesystem>
#include <stdexcept>

using State = moveit::core::RobotState;
using V6 = Eigen::Matrix<double,6,1>;
constexpr double PI = 3.14159265358979323846;
struct Target { double t; Eigen::Isometry3d T; };
std::vector<Target> read_targets() {
  std::ifstream file("generated/targets.csv");
  if (!file) throw std::runtime_error("Run prepare_workpiece.py first");
  std::string line; std::getline(file,line); std::vector<Target> result;
  while(std::getline(file,line)) {
    std::replace(line.begin(),line.end(),',',' '); std::istringstream row(line);
    double t,x,y,z,qx,qy,qz,qw;
    if(!(row>>t>>x>>y>>z>>qx>>qy>>qz>>qw)) throw std::runtime_error("Bad target row");
    Eigen::Isometry3d T=Eigen::Isometry3d::Identity();
    T.translation()=Eigen::Vector3d(x,y,z); T.linear()=Eigen::Quaterniond(qw,qx,qy,qz).normalized().toRotationMatrix();
    result.push_back({t,T});
  }
  if(result.size()<3) throw std::runtime_error("Need at least 3 targets");
  return result;
}
void add_mesh(planning_scene::PlanningScene& scene, rclcpp::Node& node) {
  auto uri="file://"+std::filesystem::absolute("generated/workpiece_mm.stl").string();
  std::unique_ptr<shapes::Mesh> mesh(shapes::createMeshFromResource(uri,Eigen::Vector3d::Constant(0.001)));
  if(!mesh) throw std::runtime_error("Cannot read workpiece mesh");
  shapes::ShapeMsg shape; shapes::constructMsgFromShape(mesh.get(),shape);
  moveit_msgs::msg::CollisionObject obj; obj.id="dataset_coupon"; obj.header.frame_id="world"; obj.operation=obj.ADD;
  obj.meshes.push_back(boost::get<shape_msgs::msg::Mesh>(shape));
  auto c=node.get_parameter("workpiece_origin").as_double_array();
  geometry_msgs::msg::Pose pose;pose.position.x=c[0];pose.position.y=c[1];pose.position.z=c[2];pose.orientation.w=1;
  obj.mesh_poses.push_back(pose);
  if(!scene.processCollisionObjectMsg(obj)) throw std::runtime_error("Mesh rejected");
}
void add_table(planning_scene::PlanningScene& scene, rclcpp::Node& node) {
  auto c=node.get_parameter("table_center").as_double_array();
  auto d=node.get_parameter("table_size").as_double_array();
  moveit_msgs::msg::CollisionObject obj; obj.id="table";obj.header.frame_id="world";obj.operation=obj.ADD;
  shape_msgs::msg::SolidPrimitive shape;shape.type=shape.BOX;shape.dimensions={d[0],d[1],d[2]};
  geometry_msgs::msg::Pose pose;pose.position.x=c[0];pose.position.y=c[1];pose.position.z=c[2];pose.orientation.w=1;
  obj.primitives.push_back(shape);obj.primitive_poses.push_back(pose);
  if(!scene.processCollisionObjectMsg(obj)) throw std::runtime_error("Table rejected");
}
int main(int argc,char** argv) {
  rclcpp::init(argc,argv);
  auto node=std::make_shared<rclcpp::Node>("sr5_kinematics",rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  try {
    auto targets=read_targets();
    robot_model_loader::RobotModelLoader loader(node,"robot_description");
    auto model=loader.getModel();
    if(!model) throw std::runtime_error("No model");
    auto group=model->getJointModelGroup("rokae_arm"); auto tcp=model->getLinkModel("welding_tcp");
    if(!group || !tcp || !group->getSolverInstance()) throw std::runtime_error("No group/TCP/KDL solver");
    planning_scene::PlanningScene scene(model);add_mesh(scene,*node);add_table(scene,*node);
    auto valid=[&](State* s,const moveit::core::JointModelGroup* g,const double* q) {
      s->setJointGroupPositions(g,q);s->update();
      return s->satisfiesBounds(g) && !scene.isStateColliding(*s);
    };
    State state(model);state.setToDefaultValues();
    // First seed was found and checked for the supplied seed-42 coupon fixture.
    std::vector<std::vector<double>> seeds={{2.640762309308077,-0.03260214516182112,2.153812866008137,-2.437374646541162,-0.4660958069607337,0.5109449762633808},
      {0,.5,1.5,0,1,0},{0,1,1.5,0,1.5,0},{0,-1,-1.5,0,1,0},{PI,1,1.5,0,1,0}};
    bool found=false;
    for(const auto& seed:seeds) {
      state.setJointGroupPositions(group,seed);state.update();
      if(state.setFromIK(group,targets[0].T,"welding_tcp",.25,valid)) {found=true;break;}
    }
    for(int k=0;!found && k<40;++k) {
      state.setToRandomPositions(group);
      found=state.setFromIK(group,targets[0].T,"welding_tcp",.05,valid);
    }
    if(!found) throw std::runtime_error("No collision-free start IK for this placement");
    std::vector<V6> qs;double max_position_error=0,max_angle_error=0,max_jump=0,min_sigma=1e9,max_jac_error=0;
    std::ofstream jac("results/jacobian.csv");jac<<std::setprecision(16)<<"index,time_s";
    for(int r=0;r<6;++r)for(int c=0;c<6;++c)jac<<",J"<<r<<c;
    jac<<",sigma_min_scaled\n";
    std::ofstream fk("results/fk.csv");fk<<std::setprecision(16)<<"index,time_s,x_m,y_m,z_m,qx,qy,qz,qw,position_error_m,orientation_error_rad\n";
    const double eps=1e-6;
    for(size_t i=0;i<targets.size();++i) {
      if(i && !state.setFromIK(group,targets[i].T,"welding_tcp",.05,valid))
        throw std::runtime_error("IK failed at target "+std::to_string(i));
      state.update();V6 q;state.copyJointGroupPositions(group,q.data());
      if(i) max_jump=std::max(max_jump,(q-qs.back()).cwiseAbs().maxCoeff());
      if(i && (q-qs.back()).cwiseAbs().maxCoeff()>.05) throw std::runtime_error("IK branch jump >0.05 rad");
      qs.push_back(q);
      auto T=state.getGlobalLinkTransform(tcp);Eigen::Quaterniond quat(T.linear());
      double pe=(T.translation()-targets[i].T.translation()).norm();
      double ae=Eigen::Quaterniond(targets[i].T.linear()).angularDistance(quat);
      max_position_error=std::max(max_position_error,pe);max_angle_error=std::max(max_angle_error,ae);
      fk<<i<<','<<targets[i].t<<','<<T.translation().x()<<','<<T.translation().y()<<','<<T.translation().z()
        <<','<<quat.x()<<','<<quat.y()<<','<<quat.z()<<','<<quat.w()<<','<<pe<<','<<ae<<'\n';
      Eigen::MatrixXd J;state.getJacobian(group,tcp,Eigen::Vector3d::Zero(),J);
      // Dimensionless conditioning with characteristic length 0.919 m.
      Eigen::MatrixXd scaled=J;scaled.topRows(3)/=.919;
      double sigma=scaled.jacobiSvd().singularValues().minCoeff();min_sigma=std::min(min_sigma,sigma);
      jac<<i<<','<<targets[i].t;
      for(int r=0;r<6;++r)for(int c=0;c<6;++c)jac<<','<<J(r,c);
      jac<<','<<sigma<<'\n';
      // Test both spatial linear and angular Jacobian against central differences.
      if(i%100==0)for(int j=0;j<6;++j) {
        V6 qp=q,qm=q;qp[j]+=eps;qm[j]-=eps;
        State sp(state),sm(state);sp.setJointGroupPositions(group,qp.data());sm.setJointGroupPositions(group,qm.data());sp.update();sm.update();
        auto Tp=sp.getGlobalLinkTransform(tcp),Tm=sm.getGlobalLinkTransform(tcp);
        V6 numeric;numeric.head<3>()=(Tp.translation()-Tm.translation())/(2*eps);
        Eigen::AngleAxisd aa(Tp.linear()*Tm.linear().transpose());numeric.tail<3>()=aa.axis()*aa.angle()/(2*eps);
        max_jac_error=std::max(max_jac_error,(numeric-J.col(j)).cwiseAbs().maxCoeff());
      }
    }
    if(max_position_error>1e-4 || max_angle_error>1e-3 || max_jac_error>1e-6 || min_sigma<1e-4)
      throw std::runtime_error("FK/Jacobian/singularity acceptance failed");
    // Validation of the precise linear interpolation used by the RViz replay.
    int dense_count=0;double dense_error=0,dense_angle=0;
    for(size_t i=1;i<qs.size();++i)for(int k=0;k<=5;++k) {
      double a=k/5.;V6 q=(1-a)*qs[i-1]+a*qs[i];
      state.setJointGroupPositions(group,q.data());state.update();++dense_count;
      if(!state.satisfiesBounds(group) || scene.isStateColliding(state)) throw std::runtime_error("Dense collision/bounds failure");
      auto T=state.getGlobalLinkTransform(tcp);
      Eigen::Vector3d expected=(1-a)*targets[i-1].T.translation()+a*targets[i].T.translation();
      dense_error=std::max(dense_error,(T.translation()-expected).norm());
      dense_angle=std::max(dense_angle,Eigen::Quaterniond(T.linear()).angularDistance(Eigen::Quaterniond(targets[i].T.linear())));
    }
    if(dense_error>1e-4 || dense_angle>1e-3) throw std::runtime_error("Dense Cartesian error failure");
    std::ofstream csv("results/trajectory.csv");csv<<std::setprecision(16)<<"time_s";
    for(const auto& n:group->getVariableNames())csv<<','<<n<<"_rad";
    csv<<'\n';
    for(size_t i=0;i<qs.size();++i){csv<<targets[i].t;for(double q:qs[i])csv<<','<<q;csv<<'\n';}
    std::ofstream summary("results/summary.json");summary<<std::setprecision(15);
    summary<<"{\n\"success\":true,\n\"points\":"<<qs.size()<<",\n\"duration_s\":"<<targets.back().t
      <<",\n\"max_fk_position_error_mm\":"<<max_position_error*1000
      <<",\n\"max_fk_orientation_error_deg\":"<<max_angle_error*180/PI
      <<",\n\"max_joint_step_deg\":"<<max_jump*180/PI
      <<",\n\"min_sigma_scaled_length_0_919m\":"<<min_sigma
      <<",\n\"max_jacobian_finite_difference_error\":"<<max_jac_error
      <<",\n\"dense_samples\":"<<dense_count<<",\n\"dense_collision_samples\":0,\n\"dense_out_of_bounds_samples\":0"
      <<",\n\"dense_max_position_error_mm\":"<<dense_error*1000
      <<",\n\"dense_max_orientation_error_deg\":"<<dense_angle*180/PI<<"\n}\n";
    std::cout<<"PASS: "<<qs.size()<<" IK points, "<<dense_count<<" dense collision samples, FK error "<<max_position_error*1000<<" mm\n";
    rclcpp::shutdown();return 0;
  }catch(const std::exception& e){std::cerr<<"FAILED: "<<e.what()<<'\n';rclcpp::shutdown();return 1;}
}
