
#include <vector>
#include <fstream>
#include <iostream>
#include <random>
#include <chrono>

#include <boost/program_options.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>

#include "utils/term_msg.h"
#include "geometry/FixedVtxTetMesh.hpp"
#include "io/TetMeshReader.hpp"

std::string stiffMFile, massMFile, tetMeshFile;
double density;
bool verbose = false;

static void parse_cmd(int argc, const char* argv[])
{
    namespace po = boost::program_options;
    po::options_description genericOpt("Generic options");
    genericOpt.add_options()
            ("help,h", "display help information");
    po::options_description configOpt("Configuration");
    configOpt.add_options()
            // ("neig,n", po::value<int>(&numEigv)->default_value(200), 
                    // "Maximum number of smallest eigenvalues to compute")
            ("stiff,s", po::value<std::string>(&stiffMFile)->default_value(""),
                    "Name of the stiffness matrix file")
            ("mass,m", po::value<std::string>(&massMFile)->default_value(""),
                    "Name of the mass matrix file")
            ("tet,t", po::value<std::string>(&tetMeshFile)->default_value(""),
                    "Name of the tet mesh file")
            // ("out,o", po::value<string>(&outFile)->default_value(""),
                    // "Name of the output modes file")
            ("density,d", po::value<double>(&density)->default_value(1.),
                    "Reference density value")
            // ("fmin", po::value<double>(&freqLow)->default_value(5.),
                    // "Lowest frequency value based on the estimated density")
            // ("fmax", po::value<double>(&freqHigh)->default_value(15000.),
                    // "Highest frequency value based on the estimated density")
            ("verbose,v", "Display details");
    // use configure file to specify the option
    po::options_description cfileOpt("Configure file");
    cfileOpt.add_options()
            ("cfg-file", po::value<std::string>(), "configuration file");

    po::options_description cmdOpts;
    cmdOpts.add(genericOpt).add(configOpt).add(cfileOpt);

    po::variables_map vm;
    store(po::parse_command_line(argc, argv, cmdOpts), vm);
    if ( vm.count("cfg-file") )
    {
        std::ifstream ifs(vm["cfg-file"].as<std::string>().c_str());
        store(parse_config_file(ifs, configOpt), vm);
    }
    po::notify(vm);

    if ( vm.count("help") )
    {
        printf("Usage: %s [options] \n", argv[0]);
        printf("       This executable takes as input the stiffness and mass matrices\n");
        printf("       of a tet. mesh, and computes the eigenvectors and eigenvalues\n");
        printf("       using the eigensolvers provided in Intel MKL\n");
        std::cout << cmdOpts << std::endl;
        exit(0);
    }
    verbose = vm.count("verbose") > 0;

    if ( massMFile.empty() )
    {
        PRINT_ERROR("Specify mass matrix file\n");
        exit(1);
    }

    if ( stiffMFile.empty() ) 
    {
        PRINT_ERROR("Specify stiffness matrix file\n");
        exit(1);
    }
        if ( tetMeshFile.empty() )
    {
        PRINT_ERROR("Specify tet mesh file\n");
        exit(1);
    }

    if ( density <= 0. )
    {
        PRINT_ERROR("Density value must be positive [d=%g now]\n", density);
        exit(1);
    }

    // if ( outFile.empty() ) 
    // {
    //     PRINT_ERROR("Specify the output file\n");
    //     exit(1);
    // }

    if ( verbose )
    {
        PRINT_MSG("=============== Problem Summary ===============\n");
        PRINT_MSG("Mass Matrix:                %s\n", massMFile.c_str());
        PRINT_MSG("Stiffness Matrix:           %s\n", stiffMFile.c_str());
        PRINT_MSG("Tet Mesh:                   %s\n", tetMeshFile.c_str()); 
        // PRINT_MSG("Output file:                %s\n", outFile.c_str());
        // PRINT_MSG("# of eigenvalues est.:      %d\n", numEigv);
        PRINT_MSG("Reference density:          %g\n", density);
        PRINT_MSG("===============================================\n");
    }
}

/*
 * The matrix file only stores the lower triangle part
 */
static uint8_t read_csc_dmatrix(const char* file, 
  std::vector<int>& ptrrow,
  std::vector<int>& idxcol,
  std::vector<double>& data,
  int& nrow, int& ncol) {

  using namespace std;

  ifstream fin(file, ios::binary);
  if ( fin.fail() )
  {
    PRINT_ERROR("read_csc_dmatrix:: Cannot open file [%s] to read\n", file);
    return 255;
  }

  uint8_t ret;
  fin.read((char *)&ret, sizeof(uint8_t));
  if ( ret != 1 )
  {
    PRINT_ERROR("read_csc_dmatrix:: The matrix data should be in double format\n");
    return 255;
  }

  int n;
  fin.read((char *)&ret, sizeof(uint8_t));
  fin.read((char *)&nrow, sizeof(int));
  fin.read((char *)&ncol, sizeof(int));
  fin.read((char *)&n,    sizeof(int));

  if ( (ret & 1) && (nrow != ncol) ) // symmetric
  {
    PRINT_ERROR("read_csc_dmatrix:: Symmetric matrix should be square\n");
    return 255;
  }

  ptrrow.resize(nrow+1);
  idxcol.resize(n);
  data.resize(n);
  fin.read((char *)(ptrrow.data()), sizeof(int)*(nrow+1));
  fin.read((char *)(idxcol.data()), sizeof(int)*n);
  fin.read((char *)(data.data()),   sizeof(double)*n);

  fin.close();
  return ret;
}

Vector3d centroid(const FixedVtxTetMesh<double>& tmesh) {
  double n = tmesh.num_vertices();
  Vector3d result(0.0, 0.0, 0.0);

  for (const Vector3d& v : tmesh.vertices()) {
    result += (v / n);
  }
  return result;
}

Eigen::MatrixXd getNullspace(const FixedVtxTetMesh<double>& tmesh) {
  int nVerts = tmesh.num_vertices();
  int nRows = nVerts * 3;
  Eigen::MatrixXd result(nRows, 6);

  Vector3d c_temp = centroid(tmesh);
  Eigen::Map<Eigen::Vector3d> c(c_temp);

  for (int i=0; i<nVerts; i++) {
    result.block<3,3>(3*i, 0).setIdentity();

    Vector3d v_temp = tmesh.vertex(i);
    Eigen::Map<Eigen::Vector3d> v(v_temp);
    Eigen::Vector3d ctov(v - c);

    result.block<3,1>(3*i, 3) = Eigen::Vector3d::UnitX().cross(ctov);
    result.block<3,1>(3*i, 4) = Eigen::Vector3d::UnitY().cross(ctov);
    result.block<3,1>(3*i, 5) = Eigen::Vector3d::UnitZ().cross(ctov);
  }

  result.colwise().normalize();
  return result;
}


struct SparseData {
protected:
  int nrow, ncol;
  std::vector<int> outerPtr;
  std::vector<int> innerPtr;
  std::vector<double> data;
  Eigen::Map<Eigen::SparseMatrix<double>> map;

public:
  SparseData(const char* file) : map(0, 0, 0, nullptr, nullptr, nullptr) {
    uint8_t result = read_csc_dmatrix(file, outerPtr, innerPtr, data, nrow, ncol);
    if (result == 255) throw;
    for (int i  = 0; i < outerPtr.size(); i++) {
      outerPtr[i]--;
    }
    for (int i=0; i < innerPtr.size(); i++) {
      innerPtr[i]--;
    }
    map = Eigen::Map<Eigen::SparseMatrix<double>>(nrow, ncol, data.size(), outerPtr.data(), innerPtr.data(), data.data());
  }

  const Eigen::Map<Eigen::SparseMatrix<double>>& getMap() const {
    return map;
  }

  int rows() const { return nrow; }
  int cols() const { return ncol; }

  Eigen::VectorXd diagonal() const {
    int n = rows();
    Eigen::VectorXd d(n);
    for (int i=0; i<n; i++) {
      if (outerPtr[i] == outerPtr[i+1]) {
        d(i) = 0.0;
      } else if (innerPtr[outerPtr[i]] == i) {
        d(i) = data[outerPtr[i]];
      } else {
        d(i) = 0.0;
      }
    }
    return d;
  }
};

#include <Eigen/SparseCholesky>

int checkNEVs(const SparseData& K, const SparseData& M, const double largestEV, const double largestRelError) {
  Eigen::SparseMatrix<double> A = K.getMap();
  A -= ((1.0 + largestRelError) * largestEV) * M.getMap();
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(A);
  return (ldlt.vectorD().array() < 0.0).count();
}

template<class KSolver>
Eigen::VectorXd subspaceIteration(const SparseData& K, const SparseData& M, Eigen::MatrixXd& Q, const Eigen::ArrayXd& tol,
  const int maxIters = 20, const double scale = 1.0, const double shift = -1.0) {
  using namespace Eigen;

  const int p = tol.rows();
  const int k = Q.cols();
  const int n = M.rows();
  assert(p < k);

  SparseMatrix<double> Kt = scale * K.getMap();
  Kt -= shift * M.getMap();

  MatrixXd Qk(n,k), Kk(k,k), Mk(k,k);
  ArrayXd lambda = ArrayXd::Zero(p), prevLambda(p), error(p);
  GeneralizedSelfAdjointEigenSolver<MatrixXd> gevd(k);

  KSolver Ktinv(Kt);
  if (verbose) std::cout << "Ktinv initialized" << std::endl;
  for (int iter = 0; iter<maxIters; iter++) {
    if (iter == 0) {
      Qk.noalias() = Ktinv.solve(Q);
      Kk.noalias() = Q.transpose() * Qk;
    } else {
      Qk.noalias() = Ktinv.solve(M.getMap().selfadjointView<Lower>() * Q);
      Kk.noalias() = Qk.transpose() * (Kt.selfadjointView<Lower>() * Qk);
    }
    if (verbose) std::cout << "iter " << iter << ": Ktinv solve" << std::endl;
    Mk.noalias() = Qk.transpose() * (M.getMap().selfadjointView<Lower>() * Qk);
    gevd.compute(Kk, Mk);
    assert(gevd.info() == Eigen::Success);

    Q.noalias() = Qk * gevd.eigenvectors();
    if (verbose) std::cout << "iter " << iter << ": gevd solve" << std::endl;

    // Check tolerance
    
    // if (iter > 0) {
    //   lambda = gevd.eigenvalues().head(p).array().square();
    //   for (int i=0; i<p; i++) {
    //     lambda(i) /= gevd.eigenvectors().col(i).dot(gevd.eigenvectors().col(i));
    //   }
    //   lambda = (1.0 - lambda).sqrt();
    //   if (verbose) std::cout << "iter " << iter << ": tol: " << lambda.transpose() << std::endl;

    //   if (verbose) {
    //     Eigen::VectorXd ev2(p);
    //     for (int i=0; i<p; i++) {
    //       ev2(i) = gevd.eigenvectors().col(i).dot(gevd.eigenvectors().col(i));
    //     }

    //     std::cout << "lambda: " << gevd.eigenvalues().head(p).transpose() << std::endl;
    //     std::cout << "ev2:    " << ev2.transpose() << std::endl;
    //   }

    //   if ((lambda <= tol).all()) break;
    // }

    prevLambda = lambda;
    lambda = gevd.eigenvalues().head(p).array();
    if (iter > 1) {
      error = (1.0 - (prevLambda / lambda)).square();
      if (verbose) std::cout << "iter " << iter << ": error: " << error.transpose() << std::endl;
      if ((error <= tol).all()) {
        int nEVs = checkNEVs(K, M, (lambda(p-1) + shift) / scale, error(p-1));
        if (verbose) std::cout << "Converged; expected " << nEVs << " evs, found " << p << std::endl;
        break;
      }
    }
  }

  return ((gevd.eigenvalues().head(p).array() + shift) / scale).matrix();
}

// double jnd(const double freq) {
//   if (freq <= 500.0) {
//     return 3.0;
//   }
//   if (freq < 1000.0) {
//     return (freq - 500.0) / 500.0 * 3.0 + 3.0;
//   }
//   return freq * 0.06;
// }

bool converged(const Eigen::ArrayXd& lambda, const Eigen::ArrayXd& error) {
  const Eigen::ArrayXd thresholds(lambda.sqrt() * 0.5 * M_1_PI);
  auto jnd = [] (const double freq) {
    if (freq <= 500.0) { // (500 * 2 * pi)^2 = 9869604.4012
      return 3.0; // (3 * 2 * pi)^2 = 355.30575846
    }
    if (freq < 1000.0) { // (1000 * 2 * pi)^2 = 39478417.605
      return (freq - 500.0) / 500.0 * 3.0 + 3.0;
    }
    return freq * 0.06;
  };
  return (error.abs() < thresholds.unaryExpr(jnd)).all(); // FIXME: relative error? on EVs?
}


// #include "io/TetMeshWriter.hpp"

int main(int argc, char const *argv[]) {

  parse_cmd(argc, argv);

  FixedVtxTetMesh<double> mesh;
  FV_TetMeshLoader_Double::load_mesh(tetMeshFile.c_str(), mesh);

  const double massScale = 1.0 / density;
  const double stiffnessScale = 1e-8;

  Eigen::MatrixXd R = getNullspace(mesh);

  SparseData M(massMFile.c_str());
  SparseData K(stiffMFile.c_str());

  int n = M.rows();
  int p = 10;
  // int p = 2;
  int q = std::max(p + 8, 2 * p) + R.cols();
  // int q = 9; 
  Eigen::MatrixXd Q(n, q);
  Q.leftCols(R.cols()) = R;

  std::default_random_engine generator;
  std::normal_distribution<double> dist(0.0, 1.0);
  auto normal = [&] () { return dist(generator); };

  // option: random cols
  // Q.rightCols(q - R.cols()) = Eigen::MatrixXd::NullaryExpr(n, q - R.cols(), normal);

  // option: Bathe suggestion
  Eigen::VectorXd Mdiag = M.diagonal();
  Eigen::VectorXd Kdiag = K.diagonal();

  int colIndex = R.cols();
  Q.col(colIndex) = Mdiag;
  colIndex++;

  int numecols = q - R.cols() - 2;
  Eigen::ArrayXi indices = Eigen::ArrayXi::LinSpaced(numecols, 0, numecols-1);
  Eigen::ArrayXd ratios = Kdiag.head(numecols).array() / Mdiag.head(numecols).array();
  int maxIndex = 0;
  double maxValue = ratios.maxCoeff(&maxIndex);
  for (int i=numecols; i<n; i++) {
    double r = Kdiag(i) / Mdiag(i);
    if (r < maxValue) {
      ratios(maxIndex) = r;
      indices(maxIndex) = i;
      maxValue = ratios.maxCoeff(&maxIndex);
    }
  }
  for (int i=0; i<numecols; i++, colIndex++) {
    Q.col(colIndex) = Eigen::VectorXd::Unit(n, indices(i));
  }
  Q.col(colIndex) = Eigen::VectorXd::NullaryExpr(n, normal);


  Q.colwise().normalize();

  // std::cout << "QTQ:\n" << (Q.transpose() * Q) << "\n\n"; 

  Eigen::ArrayXd tol = Eigen::ArrayXd::Constant(p+R.cols(), 1e-4);

  auto start = std::chrono::system_clock::now();
  Eigen::VectorXd evs = subspaceIteration<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>(K, M, Q, tol, 20, stiffnessScale);
  auto end = std::chrono::system_clock::now();

  std::cout << "subspaceIteration time: " << (std::chrono::duration<double>(end - start)).count() << std::endl;
  std::cout << "Evs:" << std::endl << evs << std::endl << std::endl;

  // Dense, full solve
  Eigen::MatrixXd Md = M.getMap().toDense();
  Eigen::MatrixXd Kd = K.getMap().toDense();

  start = std::chrono::system_clock::now();
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> gevd(Kd, Md, Eigen::EigenvaluesOnly);
  end = std::chrono::system_clock::now();

  std::cout << "Dense GSAES time: " << (std::chrono::duration<double>(end - start)).count() << std::endl;
  std::cout << "Ref Evs:" << std::endl << gevd.eigenvalues().head(p+R.cols()) << std::endl;

  // FixedVtxTetMesh<double> mesh;
  // mesh.add_vertex(Point3d(0.0, 0.0, 0.0));
  // mesh.add_vertex(Point3d(1.0, 0.0, 0.0));
  // mesh.add_vertex(Point3d(0.0, 1.0, 0.0));
  // mesh.add_vertex(Point3d(0.0, 0.0, 1.0));
  // mesh.add_tet(0, 1, 2, 3);
  // mesh.init();
  // FV_TetMeshWriter_Double::write_mesh(argv[1], mesh);

  return 0;
}
