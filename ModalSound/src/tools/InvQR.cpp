
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


std::string stiffMFile, massMFile, tetMeshFile, outFile;
double density, tolerance;
bool verbose = false;
bool ref = false;
int numEigv;

int i_initialSubspace, i_convTest, i_ksolver;

enum InitialSubspace {
  BatheSuggestion = 0,
  Random = 1,
} initialSubspace;

const char* initialSubspaceName(InitialSubspace is) {
  switch (is) {
  case InitialSubspace::BatheSuggestion:
    return "Bathe";
  case InitialSubspace::Random:
    return "Random";
  default:
    return "";
  }
}

enum ConvergenceTest {
  Bathe = 0,
  JND = 1,
  Trace = 2,
  Rayleigh = 3,
} convTest;

const char* convgergenceTestName(ConvergenceTest ct) {
  switch (ct) {
  case ConvergenceTest::Bathe:
    return "Bathe";
  case ConvergenceTest::JND:
    return "JND";
  case ConvergenceTest::Trace:
    return "Trace";
  case ConvergenceTest::Rayleigh:
    return "Rayleigh";
  default:
    return "";
  }
}

enum KSolver {
  LDLT = 0,
  PCG = 1,
} ksolver;

const char* kSolverName(KSolver ks) {
  switch (ks) {
  case KSolver::LDLT:
    return "LDLT";
  case KSolver::PCG:
    return "PCG";
  default:
    return "";
  }
}

static void parse_cmd(int argc, const char* argv[])
{
    namespace po = boost::program_options;
    po::options_description genericOpt("Generic options");
    genericOpt.add_options()
            ("help,h", "display help information");
    po::options_description configOpt("Configuration");
    configOpt.add_options()
            ("neig,n", po::value<int>(&numEigv)->default_value(16), 
                    "Maximum number of smallest eigenvalues to compute")
            ("stiff,s", po::value<std::string>(&stiffMFile)->default_value(""),
                    "Name of the stiffness matrix file")
            ("mass,m", po::value<std::string>(&massMFile)->default_value(""),
                    "Name of the mass matrix file")
            ("tet,t", po::value<std::string>(&tetMeshFile)->default_value(""),
                    "Name of the tet mesh file")
            ("out,o", po::value<std::string>(&outFile)->default_value(""),
                    "Name of the output modes file")
            ("density,d", po::value<double>(&density)->default_value(1.),
                    "Reference density value")
            ("init,i", po::value<int>(&i_initialSubspace)->default_value(InitialSubspace::BatheSuggestion),
                    "Initial subspace method; 0=Bathe, 1=Random")
            ("conv,c", po::value<int>(&i_convTest)->default_value(ConvergenceTest::JND),
                    "Convgergence test; 0=Bathe, 1=JND, 2=Trace, 3=Rayleigh")
            ("tol,e", po::value<double>(&tolerance)->default_value(0.006),
                    "Tolerance for convergence")
            ("ksolve,k", po::value<int>(&i_ksolver)->default_value(KSolver::LDLT),
                    "Stiffness system solver; 0=LDLT, 1=PCG")
            ("ref,r", "Compute reference solution")
            ("verbose,v", "Display details");
    // use configure file to specify the option
    po::options_description cfileOpt("Configure file");
    cfileOpt.add_options()
            ("cfg-file", po::value<std::string>(), "configuration file");

    po::options_description cmdOpts;
    cmdOpts.add(genericOpt).add(configOpt).add(cfileOpt);

    po::variables_map vm;
    store(po::parse_command_line(argc, argv, cmdOpts), vm);
    if (vm.count("cfg-file")) {
      std::ifstream ifs(vm["cfg-file"].as<std::string>().c_str());
      store(parse_config_file(ifs, configOpt), vm);
    }
    po::notify(vm);

    if (vm.count("help")) {
      printf("Usage: %s [options] \n", argv[0]);
      printf("       This executable takes as input the stiffness and mass matrices\n");
      printf("       of a tet. mesh, and computes the eigenvectors and eigenvalues\n");
      printf("       using the subspace iteration method.\n");
      std::cout << cmdOpts << std::endl;
      exit(0);
    }
    verbose = vm.count("verbose") > 0;

    if (massMFile.empty()) {
      PRINT_ERROR("Specify mass matrix file\n");
      exit(1);
    }

    if (stiffMFile.empty()) {
      PRINT_ERROR("Specify stiffness matrix file\n");
      exit(1);
    }

    if (tetMeshFile.empty()) {
      PRINT_ERROR("Specify tet mesh file\n");
      exit(1);
    }

    if (density <= 0.0) {
      PRINT_ERROR("Density value must be positive [d=%g now]\n", density);
      exit(1);
    }

    if (numEigv <= 6) {
      PRINT_ERROR("numEigv must be greater than 6");
      exit(1);
    }

    if (i_initialSubspace > 1) {
      PRINT_ERROR("Initial subspace method must be less than 2");
      exit(1);
    } else {
      initialSubspace = (InitialSubspace) i_initialSubspace;
    }

    if (i_convTest > 3) {
      PRINT_ERROR("Convergence test method must be less than 4");
      exit(1);
    } else {
      convTest = (ConvergenceTest) i_convTest;
    }

    if (i_ksolver > 1) {
      PRINT_ERROR("ksolver must be less than 2");
      exit(1);
    } else {
      ksolver = (KSolver) i_ksolver;
    }

    ref = vm.count("ref") > 0;

    if (verbose) {
      PRINT_MSG("=============== Problem Summary ===============\n");
      PRINT_MSG("Mass Matrix:                %s\n", massMFile.c_str());
      PRINT_MSG("Stiffness Matrix:           %s\n", stiffMFile.c_str());
      PRINT_MSG("Tet Mesh:                   %s\n", tetMeshFile.c_str());
      PRINT_MSG("Output file:                %s\n", outFile.c_str());
      PRINT_MSG("# of eigenvalues est.:      %d\n", numEigv);
      PRINT_MSG("Reference density:          %g\n", density);
      PRINT_MSG("Initial subspace method:    %s\n", initialSubspaceName(initialSubspace));
      PRINT_MSG("Convergence test:           %s\n", convgergenceTestName(convTest));
      PRINT_MSG("Convergence tolerance:      %g\n", tolerance);
      PRINT_MSG("KSolver:                    %s\n", kSolverName(ksolver));
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

/*
 * File Format:
 * <int>: size of the eigen problem
 * <int>: # of eigenvalues
 * <eigenvalues>
 * <eigenvec_1>
 * ...
 * <eigenvec_n>
 */

void write_eigenvalues(const Eigen::MatrixXd& eigenvectors, const Eigen::ArrayXd& eigenvalues,
  const char* file) {
  using namespace std;

  ofstream fout(file, std::ios::binary);
  if (!fout.good())  {
    cerr << "write_eigenvalues::Cannot open file " << file << " to write" << endl;
    return;
  }

  // size of the eigen-problem. Here square matrix is assumed.
  int nsz = eigenvectors.rows();
  fout.write((char *)&nsz, sizeof(int));
  int nev = eigenvalues.size();
  fout.write((char *)&nev, sizeof(int));

  // output eigenvalues
  for (int vid = 0; vid < nev; ++vid) {
    fout.write((const char*)&(eigenvalues.data()[vid]), sizeof(double));
    printf("ev#%3d:  %lf %lfHz\n", vid, eigenvalues(vid), sqrt(eigenvalues(vid)/density)*0.5*M_1_PI);
  }

  // output eigenvectors
  for (int vid = 0; vid < nev; ++vid) {
    fout.write((const char*)&(eigenvectors.data()[vid * nsz]), sizeof(double)*nsz);
  }

  fout.close();
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


bool convergedBathe(const Eigen::MatrixXd& Q, const Eigen::ArrayXd& lambda, const double tol) {
  Eigen::ArrayXd denom(lambda.size());
  for (int i=0; i<denom.size(); i++) {
    denom(i) = Q.col(i).dot(Q.col(i));
  }
  const Eigen::ArrayXd error = (1.0 - lambda.square() / denom).sqrt();
  if (verbose) std::cout << "error: " << error.transpose() << std::endl;
  return (error < tol).all();
}

bool convergedJND(const Eigen::ArrayXd& prevLambda, const Eigen::ArrayXd& lambda, const double tol) {
  const Eigen::ArrayXd prevFreq = prevLambda.sqrt(); // * 0.5 * M_1_PI);
  const Eigen::ArrayXd freq = lambda.sqrt(); // * 0.5 * M_1_PI);
  const Eigen::ArrayXd error = (freq - prevFreq).abs() / freq;
  if (verbose) std::cout << "error: " << error.transpose() << std::endl;
  return (error < tol).all();
}

bool convergedTrace(const Eigen::ArrayXd& prevLambda, const Eigen::ArrayXd& lambda, const double tol) {
  const double prevTrace = prevLambda.sum();
  const double trace = lambda.sum();
  const double error = std::abs(trace - prevTrace) / lambda(lambda.size()-1);
  if (verbose) std::cout << "error: " << error << std::endl;
  return error < tol;
}

bool convergedRayleigh(const Eigen::ArrayXd& lambda, const Eigen::MatrixXd& X,
  const Eigen::SparseMatrix<double>& K, const Eigen::SparseMatrix<double>& M, const double tol) {
  const Eigen::MatrixXd MX = M.selfadjointView<Eigen::Lower>() * X.leftCols(lambda.size());
  const Eigen::MatrixXd KX = K.selfadjointView<Eigen::Lower>() * X.leftCols(lambda.size());
  Eigen::ArrayXd error(lambda.size());
  for (int i=0; i<error.size(); i++) {
    error(i) = std::abs(X.col(i).dot(KX.col(i)) / X.col(i).dot(MX.col(i)) - lambda(i));
  }
  if (verbose) std::cout << "error: " << error << error.transpose() << std::endl;
  return (error < tol).all();
}

int checkNEVs(const SparseData& K, const SparseData& M, const double largestEV, const double largestRelError) {
  Eigen::SparseMatrix<double> A = K.getMap();
  A -= ((1.0 + largestRelError) * largestEV) * M.getMap();
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(A);
  return (ldlt.vectorD().array() < 0.0).count();
}

template<class KSolver>
Eigen::VectorXd subspaceIteration(const SparseData& K, const SparseData& M, Eigen::MatrixXd& X, const int p,
  const int maxIters = 20, const double scale = 1.0, const double shift = -1.0) {
  using namespace Eigen;

  const int k = X.cols();
  const int n = M.rows();
  assert(p < k);

  SparseMatrix<double> Kt = scale * K.getMap();
  Kt -= shift * M.getMap();

  MatrixXd Xk(n,k), Kk(k,k), Mk(k,k);
  ArrayXd lambda = ArrayXd::Zero(p), prevLambda(p), error(p);
  GeneralizedSelfAdjointEigenSolver<MatrixXd> gevd(k);

  KSolver Ktinv(Kt);
  // SparseMatrix<double> Ktsav = Kt.selfadjointView<Eigen::Lower>();
  // KSolver Ktinv(Ktsav);

  if (verbose) std::cout << "Ktinv initialized" << std::endl;
  for (int iter = 0; iter<maxIters; iter++) {
    if (iter == 0) {
      Xk.noalias() = Ktinv.solve(X);
      Kk.noalias() = X.transpose() * Xk;
    } else {
      Xk.noalias() = Ktinv.solve(M.getMap().selfadjointView<Lower>() * X);
      Kk.noalias() = Xk.transpose() * (Kt.selfadjointView<Lower>() * Xk);
    }
    if (verbose) std::cout << "iter " << iter << ": Ktinv solve" << std::endl;
    Mk.noalias() = Xk.transpose() * (M.getMap().selfadjointView<Lower>() * Xk);
    gevd.compute(Kk, Mk);
    assert(gevd.info() == Eigen::Success);

    X.noalias() = Xk * gevd.eigenvectors();
    if (verbose) std::cout << "iter " << iter << ": gevd solve" << std::endl;

    prevLambda = lambda;
    lambda = gevd.eigenvalues().head(p).array();
    // Check for convergence
    bool converged = false;
    if (convTest == ConvergenceTest::Bathe) {
      converged = convergedBathe(gevd.eigenvectors(), lambda, tolerance);
    } else if (convTest == ConvergenceTest::JND && iter >= 1) {
      converged = convergedJND(prevLambda, lambda, tolerance);
    } else if (convTest == ConvergenceTest::Trace && iter >= 1) {
      converged = convergedTrace(prevLambda, lambda, tolerance);
    } else if (convTest == ConvergenceTest::Rayleigh) {
      converged = convergedRayleigh(lambda, X, Kt, M.getMap(), tolerance);
    }

    if (converged) {
      int nEVs = checkNEVs(K, M, (lambda(p-1) + shift) / scale, error(p-1));
      if (verbose) std::cout << "Converged; expected " << nEVs << " evs, found " << p << std::endl;
      break;
    }
  }

  return ((gevd.eigenvalues().head(p).array() + shift) / scale).matrix();
}

int main(int argc, char const *argv[]) {

  parse_cmd(argc, argv);

  FixedVtxTetMesh<double> mesh;
  FV_TetMeshLoader_Double::load_mesh(tetMeshFile.c_str(), mesh);

  SparseData M(massMFile.c_str());
  SparseData K(stiffMFile.c_str());

  int n = M.rows();
  int k = numEigv;
  int p = std::max(k + 8, 2 * k);

  auto start = std::chrono::system_clock::now();
  Eigen::MatrixXd R = getNullspace(mesh);
  Eigen::MatrixXd Q(n, k+p);
  Q.leftCols(R.cols()) = R;

  std::default_random_engine generator;
  std::normal_distribution<double> dist(0.0, 1.0);
  auto normal = [&] () { return dist(generator); };

  Eigen::VectorXd Mdiag = M.diagonal();
  Eigen::VectorXd Kdiag = K.diagonal();

  if (initialSubspace == InitialSubspace::BatheSuggestion) {
    int colIndex = R.cols();
    Q.col(colIndex) = Mdiag;
    colIndex++;

    int numecols = Q.cols() - R.cols() - 2;
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

  } else if (initialSubspace == InitialSubspace::Random) {
    Q.rightCols(Q.cols() - R.cols()) = Eigen::MatrixXd::NullaryExpr(n, Q.cols() - R.cols(), normal);
  }

  Q.colwise().normalize();

  const double stiffnessScale = (1.0 / Kdiag.array().mean());
  if (verbose) std::cout << "stiffnessScale: " << stiffnessScale << std::endl;

  Eigen::VectorXd evs = subspaceIteration<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>(K, M, Q, p+R.cols(), 20, stiffnessScale);
  // Eigen::VectorXd evs = subspaceIteration<Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Upper|Eigen::Lower, Eigen::IncompleteCholesky<double>>>(K, M, Q, p+R.cols(), 20, stiffnessScale);
  auto end = std::chrono::system_clock::now();

  std::cout << "subspaceIteration time: " << (std::chrono::duration<double>(end - start)).count() << std::endl;
  std::cout << "Evs:" << std::endl << evs << std::endl << std::endl;

  if (!outFile.empty()) {
    write_eigenvalues(Q, evs, outFile.c_str());
  }

  if (ref) {
    // Dense solve
    Eigen::MatrixXd Md = M.getMap().toDense();
    Eigen::MatrixXd Kd = K.getMap().toDense();

    start = std::chrono::system_clock::now();
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> gevd(Kd, Md, Eigen::EigenvaluesOnly);
    end = std::chrono::system_clock::now();

    std::cout << "Dense GSAES time: " << (std::chrono::duration<double>(end - start)).count() << std::endl;
    std::cout << "Ref Evs:" << std::endl << gevd.eigenvalues().head(p+R.cols()) << std::endl;
  }

  return 0;
}
