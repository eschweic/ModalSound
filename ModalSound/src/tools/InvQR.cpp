
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
        // PRINT_MSG("Low frequency:              %g\n", freqLow);
        // PRINT_MSG("High frequency:             %g\n", freqHigh);
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
    for (int i  = 0; i  < outerPtr.size(); i++) {
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
};


class SparsePlusLowRank;

namespace Eigen {
  namespace internal {
    template<>
    struct traits<SparsePlusLowRank> : public Eigen::internal::traits<Eigen::SparseMatrix<double>> { };
  }
}

class SparsePlusLowRank : public Eigen::EigenBase<SparsePlusLowRank> {
public:
  typedef double Scalar;
  typedef double RealScalar;
  typedef int StorageIndex;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };

  Index rows() const { return M.getMap().rows(); }
  Index cols() const { return M.getMap().cols(); }

  template<typename Rhs>
  Eigen::Product<SparsePlusLowRank,Rhs,Eigen::AliasFreeProduct> operator*(const Eigen::EigenBase<Rhs>& x) const {
    return Eigen::Product<SparsePlusLowRank,Rhs,Eigen::AliasFreeProduct>(*this, x.derived());
  }


  SparsePlusLowRank(const SparseData& M, const SparseData& K, const Eigen::MatrixXd& R) : M(M), K(K), R(R), Minv(M.getMap()) {
    if (M.getMap().rows() != K.getMap().rows() || M.getMap().cols() != K.getMap().cols()) {
      PRINT_ERROR("The size of M and K must match");
      throw;
    }
    if (R.rows() != M.getMap().rows()) {
      PRINT_ERROR("R, M, and K must have the same number of rows");
      throw;
    }
  }

  const SparseData& M;
  const SparseData& K;
  const Eigen::MatrixXd& R;
  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::IncompleteCholesky<double, Eigen::Lower>> Minv;
};

namespace Eigen {
  namespace internal {
    template<typename Rhs>
    struct generic_product_impl<SparsePlusLowRank, Rhs, SparseShape, DenseShape, GemvProduct>
    : generic_product_impl_base<SparsePlusLowRank, Rhs, generic_product_impl<SparsePlusLowRank, Rhs>> {
      typedef typename Product<SparsePlusLowRank,Rhs>::Scalar Scalar;

      template<typename Dest>
      static void scaleAndAddTo(Dest& dst, const SparsePlusLowRank& lhs, const Rhs& rhs, const Scalar& alpha) {
        // This method implements "dst += alpha * lhs * rhs" inplace
        dst.noalias() += alpha * lhs.Minv.solve(lhs.K.getMap().selfadjointView<Eigen::Lower>() * rhs);
        dst.noalias() += alpha * lhs.R * (lhs.R.transpose() * rhs);
      }
    };
  }
}

typedef Eigen::ConjugateGradient<SparsePlusLowRank, Eigen::Upper|Eigen::Lower, Eigen::IdentityPreconditioner> SPLRSolver;

Eigen::VectorXd invQR(const SparsePlusLowRank& A, int k) {
  int n = A.rows();
  int maxIters = 20;

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<double> dist(0.0, 1.0);
  auto normal = [&] () { return dist(generator); };
  Eigen::MatrixXd Z = Eigen::MatrixXd::NullaryExpr(n, k, normal);
  Eigen::MatrixXd Y(n, k);
  Eigen::VectorXd S(k);
  SPLRSolver solver(A);

  for (int iter=0; iter<maxIters; iter++) {
    for (int i=0; i<k; i++) {
      Y.col(i) = solver.solve(Z.col(i));
    }
    Z = Y.colPivHouseholderQr().matrixQ();

    for (int i=0; i<k; i++) {
      S(i) = Z.col(i).dot(A.K.getMap().selfadjointView<Eigen::Lower>() * Z.col(i)) /
      Z.col(i).dot(A.M.getMap().selfadjointView<Eigen::Lower>() * Z.col(i));
    }
  }

  return S.cwiseInverse();
}


// #include "io/TetMeshWriter.hpp"

int main(int argc, char const *argv[]) {

  parse_cmd(argc, argv);

  FixedVtxTetMesh<double> mesh;
  FV_TetMeshLoader_Double::load_mesh(tetMeshFile.c_str(), mesh);

  Eigen::MatrixXd R = getNullspace(mesh);

  SparseData M(massMFile.c_str());
  SparseData K(stiffMFile.c_str());
  SparsePlusLowRank splr(M, K, R);

  // Verify MinvK + RR^T is nonsingular
  // Eigen::MatrixXd A(R.rows(), R.rows());
  // for (int i=0; i<A.cols(); i++) {
  //   A.col(i).noalias() = splr * Eigen::VectorXd::Unit(A.cols(), i);
  // }
  // std::cout << "rank: " << A.colPivHouseholderQr().rank() << " / " << R.rows() << std::endl;
  // std::cout << "A * R norm: " << (A * R).norm() << std::endl;
  // for (int i=0; i<6; i++) {
  //   std::cout << "K * R(" << i << ") norm: " << (K.getMap().selfadjointView<Eigen::Lower>() * R.col(i)).norm() << std::endl;
  // }

  std::cout << invQR(splr, 10) << std::endl;

  // Eigen::MatrixXd Md = M.getMap().toDense();
  // Md = Md.selfadjointView<Eigen::Lower>();
  // std::cout << "M:\n" << Md << "\n\n";
  // std::cout << "M rank: " << Md.colPivHouseholderQr().rank() << " / " << R.rows() << std::endl;

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
