#include <iostream>
#include "tools.h"
#include <vector>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  if(estimations.size() == 0 ||
      estimations.size() != ground_truth.size()) {
    cout << "Invalid inputs.  The estimations array must not be empty and the input arrays must be the same size.\n;";
    return rmse;
  }

  for(int i = 0; i < estimations.size(); ++i) {
    VectorXd residuals = estimations[i] - ground_truth[i];
    VectorXd squared_residuals = residuals.array() * residuals.array();
    rmse += squared_residuals;
  }

  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
  Hj << 0,0,0,0,
        0,0,0,0,
        0,0,0,0;
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // precomputed values
  float c1 = px * px + py * py;
  float c2 = sqrt(c1);
  float c3 = c1 * c2;

  double threshold = 0.1; // if px or py are smaller than the threshold, this causes inaccurate update statements.
  if(abs(px) < threshold || abs(py) < threshold) {
    Hj << 0,0,0,0,
          0,0,0,0,
          0,0,0,0;
    return Hj;
  }

  // compute the Jacobian Matrix
  Hj << px / c2, py / c2, 0, 0,
        -py / c1, px / c1, 0, 0,
        py*(vx*py - vy*px) / c3, px*(vy*px - vx*py) / c3, px / c2, py / c2;
  return Hj;
}
