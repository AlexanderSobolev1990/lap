//----------------------------------------------------------------------------------------------------------------------
///
/// \file       test_lap.cpp
/// \brief      Тестирование задач о назначениях
/// \date       18.05.22 - создан
/// \author     Соболев А.А.
///

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_lap

#define MATPLOTLIB
//#define PRINTTOTXT

#include <boost/test/unit_test.hpp>
#include <vector>
#include <set>
#include <map>
#include <armadillo>
#include <random>

#include <timing.h>
#include <lap.h>
#include <sparse.h>

#ifdef MATPLOTLIB
    #include <matplotlibcpp.h>
#endif

double drand( double dmin, double dmax )
{
    double d = static_cast<double>( rand() ) / RAND_MAX;
    return dmin + d * ( dmax - dmin );
}

bool show = false;// true;//

//----------------------------------------------------------------------------------------------------------------------
// Тестовые задачи:
//----------------------------------------------------------------------------------------------------------------------
// 1
arma::mat mat_1_dense = {
    { 7.0, 2.0, 1.0, 9.0, 4.0 },
    { 9.0, 6.0, 9.0, 6.0, 5.0 },
    { 3.0, 8.0, 3.0, 1.0, 8.0 },
    { 7.0, 9.0, 4.0, 2.0, 2.0 },
    { 8.0, 4.0, 7.0, 4.0, 8.0 }
};
arma::ivec expected_1_min = { 2, 4, 0, 3, 1 };
arma::ivec expected_1_max = { 3, 2, 4, 1, 0 };

arma::ivec expected_1_subextr_min = { 2, 0, 3, 4, 1 };
arma::ivec expected_1_subextr_max = { 3, 0, 4, 1, 2 };
//----------------------------------------------------------------------------------------------------------------------
// 2
arma::mat mat_2_dense = {
    { 93.0, 93.0, 91.0, 94.0, 99.0, 99.0, 90.0, 92.0 },
    { 96.0, 93.0, 90.0, 94.0, 98.0, 96.0, 97.0, 91.0 },
    { 96.0, 90.0, 91.0, 90.0, 92.0, 90.0, 93.0, 96.0 },
    { 93.0, 94.0, 95.0, 96.0, 97.0, 10.0, 92.0, 93.0 },
    { 94.0, 93.0, 95.0, 91.0, 90.0, 97.0, 96.0, 92.0 },
    { 94.0, 93.0, 96.0, 90.0, 93.0, 89.0, 88.0, 91.0 },
    { 94.0, 96.0, 91.0, 90.0, 95.0, 93.0, 92.0, 94.0 },
    { 93.0, 94.0, 6.0,  95.0, 91.0, 99.0, 91.0, 96.0 }
};
arma::ivec expected_2_min = { 0, 7, 1, 5, 4, 6, 3, 2 };
arma::ivec expected_2_max = { 4, 0, 7, 3, 6, 2, 1, 5 };

arma::ivec expected_2_subextr_min = expected_2_min;
arma::ivec expected_2_subextr_max = { 4, 6, 0, 3, 7, 2, 1, 5 };
//----------------------------------------------------------------------------------------------------------------------
// 3 (only max)
arma::mat mat_3_dense = {
    { 1.0, 2.0, 0.0, 0.0, 0.0, 0.0 },
    { 0.0, 3.0, 0.0, 4.0, 0.0, 0.0 },
    { 0.0, 0.0, 5.0, 6.0, 7.0, 0.0 },
    { 8.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    { 0.0, 0.0, 0.0, 0.0, 9.0, 0.0 },
    { 0.0, 0.0, 0.0, 0.0, 0.0,10.0 },
};
arma::ivec expected_3_max = { 1, 3, 2, 0, 4, 5 };
arma::ivec expected_3_subextr_max = expected_2_max;
//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_1_min )

BOOST_AUTO_TEST_CASE( test_mat_1_seqextr_min )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1.0e7;//mat_1_dense.max();
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::SequentalExtremum( mat_1_dense, SPML::LAP::TSearchParam::SP_Min, infValue, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_subextr_min, "absdiff", eps ), true );

    arma::ivec actualCOO = arma::ivec( size, arma::fill::zeros );
    double lapcostCOO;
    SPML::Sparse::CMatrixCOO coo;
    SPML::Sparse::MatrixDenseToCOO( mat_1_dense, coo );
    SPML::LAP::SequentalExtremum( coo, SPML::LAP::TSearchParam::SP_Min, infValue, resolution, actualCOO, lapcostCOO );
    BOOST_CHECK_EQUAL( arma::approx_equal( actualCOO, expected_1_subextr_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_1_jvc_dense_min )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1.0e7;
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::JVCdense( mat_1_dense, size, SPML::LAP::TSearchParam::SP_Min, infValue, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_1_mack_min )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1.0e7;
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::Mack( mat_1_dense, size, SPML::LAP::TSearchParam::SP_Min, infValue, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_1_hungarian_min )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1.0e7;
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::Hungarian( mat_1_dense, size, SPML::LAP::TSearchParam::SP_Min, infValue, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_1_max )

BOOST_AUTO_TEST_CASE( test_mat_1_seqextr_max )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1.0e7;//mat_1_dense.max();
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::SequentalExtremum( mat_1_dense, SPML::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_subextr_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_1_jvc_dense_max )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1.0e7;
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::JVCdense( mat_1_dense, size, SPML::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_1_mack_max )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1.0e7;
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::Mack( mat_1_dense, size, SPML::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_1_hungarian_max )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1.0e7;
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::Hungarian( mat_1_dense, size, SPML::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_2_min )

BOOST_AUTO_TEST_CASE( test_mat_2_seqextr_min )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1.0e7;//mat_1_dense.max();
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::SequentalExtremum( mat_2_dense, SPML::LAP::TSearchParam::SP_Min, infValue, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_subextr_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_2_jvc_dense_min )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1.0e7;
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::JVCdense( mat_2_dense, size, SPML::LAP::TSearchParam::SP_Min, infValue, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_2_mack_min )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1.0e7;
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::Mack( mat_2_dense, size, SPML::LAP::TSearchParam::SP_Min, infValue, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_2_hungarian_min )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = mat_1_dense.max();
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::Hungarian( mat_2_dense, size, SPML::LAP::TSearchParam::SP_Min, infValue, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_2_max )

BOOST_AUTO_TEST_CASE( test_mat_2_seqextr_max )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1.0e7;//mat_1_dense.max();
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::SequentalExtremum( mat_2_dense, SPML::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_subextr_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_2_jvc_dense_max )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1.0e7;
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::JVCdense( mat_2_dense, size, SPML::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_2_mack_max )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1.0e7;
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::Mack( mat_2_dense, size, SPML::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_2_hungarian_max )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1.0e7;//mat_1_dense.max();
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::Hungarian( mat_2_dense, size, SPML::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_3_max )

BOOST_AUTO_TEST_CASE( test_mat_3_jvc_sparse_max )
{
    std::vector<double> csr_val;
    std::vector<int> csr_kk;
    std::vector<int> csr_first;
    SPML::Sparse::MatrixDenseToCSR( mat_3_dense, csr_val, csr_kk, csr_first );

    unsigned size = mat_3_dense.n_cols;
    arma::ivec actual1 = arma::ivec( size, arma::fill::zeros );
    arma::ivec actual2 = arma::ivec( size, arma::fill::zeros );
    double infValue =1.0e6;
    double resolution = 1e-6;
    double lapcost1, lapcost2;
    SPML::LAP::JVCdense( mat_3_dense, size, SPML::LAP::TSearchParam::SP_Max, infValue, resolution, actual1, lapcost1 );
    SPML::LAP::JVCsparse( csr_val, csr_kk, csr_first, SPML::LAP::TSearchParam::SP_Max, infValue, resolution, actual2, lapcost2 );

    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual1, expected_3_max, "absdiff", eps ), true );
    BOOST_CHECK_EQUAL( arma::approx_equal( actual2, expected_3_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE( accordance )
{
    const int n = 128;
    arma::mat mat_JVCdense( n, n, arma::fill::randn );
    arma::mat mat_Mack( n, n, arma::fill::randn );
    arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
    arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
    double lapcostJVCdense = 0.0, lapcostMack = 0.0;//, lapcostHungarian = 0.0;
    int cycle_count = 10000;
    for( int cycle = 0; cycle < cycle_count; cycle++ ) {
        std::cout << "accordance cycle = " << cycle << "/" << cycle_count << std::endl;
        arma::arma_rng::set_seed( cycle );
        mat_JVCdense.randn();
        arma::arma_rng::set_seed( cycle );
        mat_Mack.randn();

        double infValue = 1.0e7;
        double resolution = 1e-6;
        SPML::LAP::JVCdense( mat_JVCdense, n, SPML::LAP::TSearchParam::SP_Max, infValue, resolution, actualJVCdense, lapcostJVCdense );

        SPML::LAP::Mack( mat_Mack, n, SPML::LAP::TSearchParam::SP_Max, infValue, resolution, actualMack, lapcostMack );

        BOOST_CHECK_EQUAL( lapcostJVCdense, lapcostMack );

        for( int i = 0; i < n; i++ ) {
            BOOST_CHECK_EQUAL( ( std::abs( actualJVCdense(i) - actualMack(i) ) < 1e-5 ), true );
        }
    }
}

BOOST_AUTO_TEST_CASE( cycling_jvc )
{
    // Тест проверки на зацикливание метода JVC
    const int K = 96;
    const int L = 32;
    const int N = ( K + L );
    double psi_empty = -1e6;
    double resolution = 1e-6;
    arma::mat assigncost( N, N, arma::fill::zeros ); // Полная матрица ценности (включая пустые назначения до размера k+l)
    arma::mat filledcost( K, L, arma::fill::zeros ); // Матрица ценности (без пустых назначений до размера k+l )
    arma::ivec result = arma::ivec( N, arma::fill::zeros );
    double lapcost;
    int cycle_count = 10000;
    for( int cycle = 0; cycle < cycle_count; cycle++ ) {
        std::cout << "test_LAP_JV_cycling cycle = " << ( cycle + 1 ) << "/" << cycle_count << std::endl;
        assigncost.fill( psi_empty );
        filledcost.randu();
        // Проредим матрицу filledcost: в случайных местах
        double levelCut = 0.5; // Порог прореживания
        for( int i = 0; i < K; i++ ) {
            for( int j = 0; j < L; j++ ) {
                double randomVal = drand( 0.0, 1.0 );
                if( randomVal > levelCut ) {
                    filledcost(i,j) = psi_empty; // Проредим матрицу
                }
            }
        }
        assigncost.submat( 0, 0, ( K - 1 ), ( L - 1 ) ) = filledcost;
        SPML::LAP::JVCdense( assigncost, N, SPML::LAP::TSearchParam::SP_Max, std::abs( psi_empty ), resolution,
            result, lapcost );
    }
}

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE( dense4to64 )
{
    SPML::LAP::TSearchParam sp = SPML::LAP::TSearchParam::SP_Max;
    double resolution = 1.0e-6;

    bool print = true; //false; //
    int cycle_count = 100000;

    std::mt19937 gen; ///< Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_0_1( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности

#ifdef MATPLOTLIB
    namespace plt = matplotlibcpp;
    double width = 100;
    double height = 50;

    const double in2mm = 25.4;// mm (fixed)
    const double dpi = 300;// dpi (variable)
    const double mm2px = dpi / in2mm;//
    size_t pixels_width = std::round( width * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
//    plt::title( "Время решения задачи о назначениях на плотных матрицах" );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, [мс]" );
#endif
//    std::vector<int> dimensionXlim = { 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 };
    std::vector<int> dimensionXlim;// = { 4 ... 64 };
    int startPow = 2;
    int endPow = 6;
    for( int i = startPow; i <= endPow; i++ ) {
        int value = static_cast<int>( std::pow( 2, i ) ); // Степень двойки
        dimensionXlim.push_back( value );
        if( i < endPow ) {
            int valueNext = static_cast<int>( std::pow( 2, ( i + 1 ) ) );
            dimensionXlim.push_back( value + ( ( valueNext - value ) / 4 ) );
            dimensionXlim.push_back( value + ( ( valueNext - value ) / 2 ) );
            dimensionXlim.push_back( value + 3 * ( ( valueNext - value ) / 4 ) );
        }
    }

    std::vector<double> dimensionLong( dimensionXlim.begin(), dimensionXlim.end() );
    std::vector<double> dimensionShort( dimensionXlim.begin(), dimensionXlim.end() );
    int boundN = dimensionShort.back();

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "JVCdense", dimensionLong },
        { "Mack", dimensionLong },
        { "Hungarian", dimensionShort }
    };

    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "JVCdense", {} },
        { "Mack",{} },
        { "Hungarian",{} }
    };

    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
        { "JVCdense", { { "color", "red" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCdense" } } },
        { "Mack", { { "color", "green" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "blue" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Hungarian" } } }
    };

    int counter_assign = 0;
    int counter_lapcost = 0;
    int counter_assign_sum = 0;
    int counter_lapcost_sum = 0;

    double scaleFactor = 1.0;

    for( auto &n : dimensionXlim ) {

//        arma::mat mat_JVCdense( n, n, arma::fill::randn );
//        arma::mat mat_Mack( n, n, arma::fill::randn );
//        arma::mat mat_Hungarian( n, n, arma::fill::randn );

        arma::mat mat_JVCdense( n, n, arma::fill::zeros );
        arma::mat mat_Mack( n, n, arma::fill::zeros );
        arma::mat mat_Hungarian( n, n, arma::fill::zeros );

        arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );
        double lapcostJVCdense = 0.0, lapcostMack = 0.0, lapcostHungarian = 0.0;

        std::map<std::string, SPML::Timing::CTimeKeeper> timer = {
            { "JVCdense", SPML::Timing::CTimeKeeper() },
            { "Mack", SPML::Timing::CTimeKeeper() },
            { "Hungarian", SPML::Timing::CTimeKeeper() }
        };

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }
            // old
//            arma::arma_rng::set_seed( cycle );
//            mat_JVCdense.randu();

            // new
            gen.seed( cycle );
            for( int i = 0; i < n; i++ ) {
                for( int j = 0; j < n; j++ ) {
                    double randomNum = random_0_1( gen );
                    mat_JVCdense(i,j) = randomNum * scaleFactor;
                }
            }

            /////////////////////
//            mat_JVCdense.print();
            /////////////////////

            mat_Mack = mat_JVCdense;
            mat_Hungarian = mat_JVCdense;

            double maxcost = mat_JVCdense.max() + 1.0;

            timer.at( "JVCdense" ).StartTimer();
            SPML::LAP::JVCdense( mat_JVCdense, n, sp, maxcost, resolution, actualJVCdense, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

            timer.at( "Mack" ).StartTimer();
            SPML::LAP::Mack( mat_Mack, n, sp, maxcost, resolution, actualMack, lapcostMack );
            timer.at( "Mack" ).EndTimer();

            if( n <= boundN ) {
                timer.at( "Hungarian" ).StartTimer();
                SPML::LAP::Hungarian( mat_Hungarian, n, sp, maxcost, resolution, actualHungarian, lapcostHungarian );
                timer.at( "Hungarian" ).EndTimer();
            }

            bool eq_lapcostJVCdense_lapcostMack = std::abs( lapcostJVCdense - lapcostMack ) < 1e-5;
            bool eq_lapcostJVCdense_lapcostHungarian = false;

            if( n <= boundN ) {
                eq_lapcostJVCdense_lapcostHungarian = std::abs( lapcostJVCdense - lapcostHungarian ) < 1e-5;
            }

            if( !eq_lapcostJVCdense_lapcostMack ||
                ( ( n <= boundN ) && ( !eq_lapcostJVCdense_lapcostHungarian ) ) )
            {
                counter_lapcost++;
            }

            double eps = 1e-6;
            bool eq_actualJVCdense_actualMack = arma::approx_equal( actualJVCdense, actualMack, "absdiff", eps );
            bool eq_actualJVCdense_actualHungarian = false;
            if( n <= boundN ) {
                eq_actualJVCdense_actualHungarian = arma::approx_equal( actualJVCdense, actualHungarian, "absdiff", eps );
            }
            if( !eq_actualJVCdense_actualMack ||
                ( ( n <= boundN ) && ( !eq_actualJVCdense_actualHungarian ) ) )
            {
                counter_assign++;
            }
        }/*
        if( print ) {
            for( auto &t : timer ) {
                std::cout << "timer " + t.first + " TimePerOp() = " << t.second.TimePerOp() << std::endl;
            }
        }

        for( auto &t : timer ) {
            if( ( n > boundN ) && ( t.first == "Hungarian" ) ) {
                continue;
            }
            timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() * 1.0e3 ); // мс
        }
    }
    std::cout << "counter_lapcost=" << counter_lapcost << std::endl;
    std::cout << "counter_assign=" << counter_assign << std::endl;
    */
        if( print ) {
            for( auto &t : timer ) {
                std::cout << "timer " + t.first + " TimePerOp() = " << t.second.TimePerOp() << std::endl;
            }
        }

        for( auto &t : timer ) {
            if( ( n > boundN ) && ( t.first == "Hungarian" ) ) {
                continue;
            }
            timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() * 1.0e3 ); // мс
        }
        std::cout << "counter_lapcost=" << counter_lapcost << std::endl;
        std::cout << "counter_assign=" << counter_assign << std::endl;
        counter_lapcost_sum += counter_lapcost;
        counter_assign_sum += counter_assign;
    } // for n
    std::cout << "counter_lapcost_sum=" << counter_lapcost_sum << std::endl;
    std::cout << "counter_assign_sum=" << counter_assign_sum << std::endl;


#ifdef MATPLOTLIB
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    for( auto &t : timeOfMethod ) {
        plt::plot(
            dimensionDouble.at( t.first ),
            ( t.second ), // Y
            estimated_keywords.at( t.first )
            );
    }
    plt::grid( true );
    plt::xlim( dimensionXlim.front(), dimensionXlim.back() );
    plt::ylim( 0.0, 3.0 );
    plt::legend();
    plt::save( "dense4to64.png", dpi );
    if( show ) {
        plt::show();
    }
    plt::close();
#endif

#ifdef PRINTTOTXT
    std::ofstream os;
    os.open( "dense4to64.ods", std::ofstream::out );
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    os << "N" << "\t";
    for( auto &t : timeOfMethod ) {
        os << t.first << "\t";
    }
    os << std::endl;
    for( int i = 0; i < dimensionXlim.size(); i++ ) {
        os << dimensionXlim[i] << "\t";
        for( auto &t : timeOfMethod ) {
            os << (t.second)[i] << "\t";
        }
        os << std::endl;
    }
    os.close();
#endif
}

BOOST_AUTO_TEST_CASE( dense5to50SeqExtr )
{
    SPML::LAP::TSearchParam sp = SPML::LAP::TSearchParam::SP_Max;
    double resolution = 1.0e-6;

    bool print = true; //false; //
    int cycle_count = 1000;

#ifdef MATPLOTLIB
    namespace plt = matplotlibcpp;
    double width = 100;
    double height = 50;

    const double in2mm = 25.4;// mm (fixed)
    const double dpi = 300;// dpi (variable)
    const double mm2px = dpi / in2mm;//
    size_t pixels_width = std::round( width * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
//    plt::title( "Время решения задачи о назначениях на плотных матрицах" );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, [мс]" );
#endif
    std::vector<int> dimensionXlim = { 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 };
    std::vector<double> dimensionLong( dimensionXlim.begin(), dimensionXlim.end() );
    std::vector<double> dimensionShort( dimensionXlim.begin(), dimensionXlim.end() );
    int boundN = dimensionShort.back();

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "SeqExtr", dimensionLong },
        { "JVCdense", dimensionLong },
        { "Mack", dimensionLong },
        { "Hungarian", dimensionShort }
    };

    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "SeqExtr", {} },
        { "JVCdense", {} },
        { "Mack",{} },
        { "Hungarian",{} }
    };

    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
        { "SeqExtr", { { "color", "orange" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "SeqExtr" } } },
        { "JVCdense", { { "color", "red" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCdense" } } },
        { "Mack", { { "color", "green" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "blue" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Hungarian" } } }
    };

    int counter_assign = 0;
    int counter_lapcost = 0;

    for( auto &n : dimensionXlim ) {
        arma::mat mat_SeqExtr( n, n, arma::fill::randn );
        arma::mat mat_JVCdense( n, n, arma::fill::randn );
        arma::mat mat_Mack( n, n, arma::fill::randn );
        arma::mat mat_Hungarian( n, n, arma::fill::randn );

        arma::ivec actualSeqExtr = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );
        double lapcostSeqExtr = 0.0, lapcostJVCdense = 0.0, lapcostMack = 0.0, lapcostHungarian = 0.0;

        std::map<std::string, SPML::Timing::CTimeKeeper> timer = {
            { "SeqExtr", SPML::Timing::CTimeKeeper() },
            { "JVCdense", SPML::Timing::CTimeKeeper() },
            { "Mack", SPML::Timing::CTimeKeeper() },
            { "Hungarian", SPML::Timing::CTimeKeeper() }
        };

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }
            arma::arma_rng::set_seed( cycle );
            mat_JVCdense.randu();

            /////////////////////
//            mat_JVCdense.print();
            /////////////////////

            mat_SeqExtr = mat_JVCdense;
            mat_Mack = mat_JVCdense;
            mat_Hungarian = mat_JVCdense;

            double maxcost = mat_JVCdense.max() + 1.0;
//            double maxcost = maxvalue;

            timer.at( "JVCdense" ).StartTimer();
            SPML::LAP::JVCdense( mat_JVCdense, n, sp, maxcost, resolution, actualJVCdense, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

            timer.at( "SeqExtr" ).StartTimer();
            SPML::LAP::SequentalExtremum( mat_SeqExtr, sp, maxcost, resolution, actualSeqExtr, lapcostSeqExtr );
            timer.at( "SeqExtr" ).EndTimer();

            timer.at( "Mack" ).StartTimer();
            SPML::LAP::Mack( mat_Mack, n, sp, maxcost, resolution, actualMack, lapcostMack );
            timer.at( "Mack" ).EndTimer();

            if( n <= boundN ) {
                timer.at( "Hungarian" ).StartTimer();
                SPML::LAP::Hungarian( mat_Hungarian, n, sp, maxcost, resolution, actualHungarian, lapcostHungarian );
                timer.at( "Hungarian" ).EndTimer();
            }

            bool eq_lapcostJVCdense_lapcostMack = std::abs( lapcostJVCdense - lapcostMack ) < 1e-5;
            bool eq_lapcostJVCdense_lapcostHungarian = false;

            if( n <= boundN ) {
                eq_lapcostJVCdense_lapcostHungarian = std::abs( lapcostJVCdense - lapcostHungarian ) < 1e-5;
            }

            if( !eq_lapcostJVCdense_lapcostMack ||
                ( ( n <= boundN ) && ( !eq_lapcostJVCdense_lapcostHungarian ) ) )
            {
                counter_lapcost++;
            }

            double eps = 1e-6;
            bool eq_actualJVCdense_actualMack = arma::approx_equal( actualJVCdense, actualMack, "absdiff", eps );
            bool eq_actualJVCdense_actualHungarian = false;
            if( n <= boundN ) {
                eq_actualJVCdense_actualHungarian = arma::approx_equal( actualJVCdense, actualHungarian, "absdiff", eps );
            }
            if( !eq_actualJVCdense_actualMack ||
                ( ( n <= boundN ) && ( !eq_actualJVCdense_actualHungarian ) ) )
            {
                counter_assign++;
            }
        }
        if( print ) {
            for( auto &t : timer ) {
                std::cout << "timer " + t.first + " TimePerOp() = " << t.second.TimePerOp() << std::endl;
            }
        }

        for( auto &t : timer ) {
            if( ( n > boundN ) && ( t.first == "Hungarian" ) ) {
                continue;
            }
            timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() * 1.0e3 ); // мс
        }
    }
    std::cout << "counter_lapcost=" << counter_lapcost << std::endl;
    std::cout << "counter_assign=" << counter_assign << std::endl;
#ifdef MATPLOTLIB
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    for( auto &t : timeOfMethod ) {
        plt::plot(
            dimensionDouble.at( t.first ),
            ( t.second ), // Y
            estimated_keywords.at( t.first )
            );
    }
    plt::grid( true );
    plt::xlim( dimensionXlim.front(), dimensionXlim.back() );
    plt::legend();
    plt::save( "dense5to50SeqExtr.png", dpi );
    if( show ) {
        plt::show();
    }
    plt::close();
#endif

#ifdef PRINTTOTXT
    std::ofstream os;
    os.open( "dense5to50.ods", std::ofstream::out );
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    os << "N" << "\t";
    for( auto &t : timeOfMethod ) {
        os << t.first << "\t";
    }
    os << std::endl;
    for( int i = 0; i < dimensionXlim.size(); i++ ) {
        os << dimensionXlim[i] << "\t";
        for( auto &t : timeOfMethod ) {
            os << (t.second)[i] << "\t";
        }
        os << std::endl;
    }
    os.close();
#endif
}

BOOST_AUTO_TEST_CASE( dense100to1000 )
{
    SPML::LAP::TSearchParam sp = SPML::LAP::TSearchParam::SP_Max;
    double resolution = 1.0e-6;

    bool print = true; //false; //
    int cycle_count = 10;
#ifdef MATPLOTLIB
    namespace plt = matplotlibcpp;
    double width = 100;
    double height = 50;

    const double in2mm = 25.4;// mm (fixed)
    const double dpi = 300;// dpi (variable)
    const double mm2px = dpi / in2mm;//
    size_t pixels_width = std::round( width * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
//    plt::title( "Время решения задачи о назначениях на плотных матрицах" );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, [мс]" );
#endif
    std::vector<int> dimensionXlim =        { 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000 };
    std::vector<double> dimensionLongLong = { 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000 };
    std::vector<double> dimensionLong =     { 100, 150, 200, 250, 300, 350, 400, 450 };
    std::vector<double> dimensionShort =    { 100, 150, 200, 250, 300 };

    int boundN = dimensionShort.back();
    int boundN2 = dimensionLong.back();

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "JVCdense", dimensionLongLong },
        { "Mack", dimensionLong },
        { "Hungarian", dimensionShort }
    };

    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "JVCdense", {} },
        { "Mack",{} },
        { "Hungarian",{} }
    };

    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
        { "JVCdense", { { "color", "red" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCdense" } } },
        { "Mack", { { "color", "green" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "blue" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Hungarian" } } }
    };

    int counter_assign = 0;
    int counter_lapcost = 0;

    for( auto &n : dimensionXlim ) {

        arma::mat mat_JVCdense( n, n, arma::fill::randn );
        arma::mat mat_Mack( n, n, arma::fill::randn );
        arma::mat mat_Hungarian( n, n, arma::fill::randn );
        arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );
        double lapcostJVCdense = 0.0, lapcostMack = 0.0, lapcostHungarian = 0.0;

        std::map<std::string, SPML::Timing::CTimeKeeper> timer = {
            { "JVCdense", SPML::Timing::CTimeKeeper() },
            { "Mack", SPML::Timing::CTimeKeeper() },
            { "Hungarian", SPML::Timing::CTimeKeeper() }
        };

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }
            arma::arma_rng::set_seed( cycle );
            mat_JVCdense.randu();

            /////////////////////
//            mat_JVCdense.print();
            /////////////////////

            mat_Mack = mat_JVCdense;
            mat_Hungarian = mat_JVCdense;

            double maxcost = mat_JVCdense.max() + 1.0;
//            double maxcost = maxvalue;

            timer.at( "JVCdense" ).StartTimer();
            SPML::LAP::JVCdense( mat_JVCdense, n, sp, maxcost, resolution, actualJVCdense, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

            if( n <= boundN2 ) {
                timer.at( "Mack" ).StartTimer();
                SPML::LAP::Mack( mat_Mack, n, sp, maxcost, resolution, actualMack, lapcostMack );
                timer.at( "Mack" ).EndTimer();
            }

            if( n <= boundN ) {
                timer.at( "Hungarian" ).StartTimer();
                SPML::LAP::Hungarian( mat_Hungarian, n, sp, maxcost, resolution, actualHungarian, lapcostHungarian );
                timer.at( "Hungarian" ).EndTimer();
            }

            bool eq_lapcostJVCdense_lapcostMack = false;
            if( n <= boundN2 ) {
                eq_lapcostJVCdense_lapcostMack = std::abs( lapcostJVCdense - lapcostMack ) < 1e-5;
            }

            bool eq_lapcostJVCdense_lapcostHungarian = false;
            if( n <= boundN ) {
                eq_lapcostJVCdense_lapcostHungarian = std::abs( lapcostJVCdense - lapcostHungarian ) < 1e-5;
            }

            if( ( ( n <= boundN2 ) && ( !eq_lapcostJVCdense_lapcostMack ) ) ||
                ( ( n <= boundN )  && ( !eq_lapcostJVCdense_lapcostHungarian ) ) )
            {
                counter_lapcost++;
            }

            double eps = 1e-6;

            bool eq_actualJVCdense_actualMack = false;
            if( n <= boundN2 ) {
                eq_actualJVCdense_actualMack = arma::approx_equal( actualJVCdense, actualMack, "absdiff", eps );
            }

            bool eq_actualJVCdense_actualHungarian = false;
            if( n <= boundN ) {
                eq_actualJVCdense_actualHungarian = arma::approx_equal( actualJVCdense, actualHungarian, "absdiff", eps );
            }

            if( ( ( n <= boundN2 ) && ( !eq_actualJVCdense_actualMack ) ) ||
                ( ( n <= boundN )  && ( !eq_actualJVCdense_actualHungarian ) ) )
            {
                counter_assign++;
            }
        }
        if( print ) {
            for( auto &t : timer ) {
                std::cout << "timer " + t.first + " TimePerOp() = " << t.second.TimePerOp() << std::endl;
            }
        }

        for( auto &t : timer ) {
            if( ( ( n <= boundN2 ) && ( t.first == "Mack" ) ) ||
                ( ( n <= boundN )  && ( t.first == "Hungarian" ) ) ||
                ( t.first == "JVCdense" )
                )
            {
                timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() * 1.0e3 ); // мс
            }
        }
    }
    std::cout << "counter_lapcost=" << counter_lapcost << std::endl;
    std::cout << "counter_assign=" << counter_assign << std::endl;
#ifdef MATPLOTLIB
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    for( auto &t : timeOfMethod ) {
        plt::plot(
            dimensionDouble.at( t.first ),
            ( t.second ), // Y
            estimated_keywords.at( t.first )
            );
    }
    plt::grid( true );
    plt::xlim( dimensionXlim.front(), dimensionXlim.back() );
    plt::legend();
    plt::save( "dense100to1000.png", dpi );
    if( show ) {
        plt::show();
    }
    plt::close();
#endif

#ifdef PRINTTOTXT
    std::ofstream os;
    os.open( "dense100to1000.ods", std::ofstream::out );
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    os << "N" << "\t";
    for( auto &t : timeOfMethod ) {
        os << t.first << "\t";
    }
    os << std::endl;
    for( int i = 0; i < dimensionXlim.size(); i++ ) {
        os << dimensionXlim[i] << "\t";
        for( auto &t : timeOfMethod ) {
            std::string method = t.first;
            double value = 0.0; //(t.second)[i];
            if( i < ( ( t.second ).size() ) ) {
                value = ( t.second )[i];
            }
            os << value << "\t";
        }
        os << std::endl;
    }
    os.close();
#endif
}

BOOST_AUTO_TEST_CASE( sparse4to64 )
{
    SPML::LAP::TSearchParam sp = SPML::LAP::TSearchParam::SP_Max;
    double resolution = 1.0e-7;

    bool print = true; //false; //
    int cycle_count = 100000; //1;//

    std::mt19937 gen; ///< Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_0_1( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности
    std::uniform_real_distribution<double> random_double_probability( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности

#ifdef MATPLOTLIB
    namespace plt = matplotlibcpp;    
    double width = 100;
    double height = 50;

    const double in2mm = 25.4;// mm (fixed)
    const double dpi = 300;// dpi (variable)
    const double mm2px = dpi / in2mm;//
    size_t pixels_width = std::round( width * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
//    plt::title( "Время решения задачи о назначениях на разреженных матрицах" );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, [мс]" );
#endif

//    std::vector<int> dimensionXlim = { 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 };
    std::vector<int> dimensionXlim;// = { 4, 8, 16, 32, 64 };
    int startPow = 2;
    int endPow = 6;//4;//3;//
    for( int i = startPow; i <= endPow; i++ ) {
        int value = static_cast<int>( std::pow( 2, i ) ); // Степень двойки
        dimensionXlim.push_back( value );
        if( i < endPow ) {
            int valueNext = static_cast<int>( std::pow( 2, ( i + 1 ) ) );
            dimensionXlim.push_back( value + ( ( valueNext - value ) / 4 ) );
            dimensionXlim.push_back( value + ( ( valueNext - value ) / 2 ) );
            dimensionXlim.push_back( value + 3 * ( ( valueNext - value ) / 4 ) );
        }
    }

    std::vector<double> dimensionLong( dimensionXlim.begin(), dimensionXlim.end() );
    std::vector<double> dimensionShort( dimensionXlim.begin(), dimensionXlim.end() );

    int boundN = dimensionShort.back();

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "JVCdense", dimensionLong },
        { "JVCsparse", dimensionLong },
        { "Mack", dimensionLong },
        { "Hungarian", dimensionShort }
    };

    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "JVCdense", {} },
        { "JVCsparse", {} },
        { "Mack",{} },
        { "Hungarian",{} }
    };

    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
        { "JVCdense", { { "color", "red" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCdense" } } },
        { "JVCsparse", { { "color", "magenta" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCsparse" } } },
        { "Mack", { { "color", "green" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "blue" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Hungarian" } } }
    };

    int counter_assign = 0;
    int counter_lapcost = 0;
    int counter_assign_sum = 0;
    int counter_lapcost_sum = 0;

    double scaleFactor = 1.0;

    for( auto &n : dimensionXlim ) {
        counter_assign = 0;
        counter_lapcost = 0;

        std::uniform_int_distribution<int> random_uint_0_n( 0, ( n - 1 ) );

        SPML::Sparse::CMatrixCSR mat_JVCsparse;
        SPML::Sparse::CMatrixCSR mat_JVCsparse2N;

        arma::mat mat_JVCdense( n, n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse( n, n, arma::fill::zeros );
        arma::mat mat_Mack( n, n, arma::fill::zeros );
        arma::mat mat_Hungarian( n, n, arma::fill::zeros );

        arma::mat mat_JVCdense2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Mack2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Hungarian2N( 2*n, 2*n, arma::fill::zeros );

        arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualJVCsparse = arma::ivec( n, arma::fill::zeros );        
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );

        arma::ivec actualJVCdense2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualJVCsparse2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualMack2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualHungarian2N = arma::ivec( 2*n, arma::fill::zeros );

        double lapcostJVCdense = 0.0, lapcostJVCsparse = 0.0, lapcostMack = 0.0, lapcostHungarian = 0.0;

        std::map<std::string, SPML::Timing::CTimeKeeper> timer = {
            { "JVCdense", SPML::Timing::CTimeKeeper() },
            { "JVCsparse", SPML::Timing::CTimeKeeper() },
            { "Mack", SPML::Timing::CTimeKeeper() },
            { "Hungarian", SPML::Timing::CTimeKeeper() }
        };

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }            
            // old
//            arma::arma_rng::set_seed( cycle );
//            mat_JVCdense.randu();// 0..1

            // new
            gen.seed( cycle );
            for( int i = 0; i < n; i++ ) {
                for( int j = 0; j < n; j++ ) {
                    double randomNum = random_0_1( gen );
                    mat_JVCdense(i,j) = randomNum * scaleFactor;
                }
            }

            mat_JVCdenseForSparse = mat_JVCdense;

//            bool doSparse = false;
            bool doSparse = true;

            double infValue = 1.0e7;
            double bigValue = -1.0e6;
            double halfBigValue = -1.0e5;

            if( doSparse ) {
                // Проредим матрицу в случайных местах
                double levelCut = 0.02; // Порог прореживания
                for( int i = 0; i < n; i++ ) {
                    int zeros_in_row = 0;
                    for( int j = 0; j < n; j++ ) {
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        if( randomDouble > levelCut ) {
                            mat_JVCdense(i, j) = bigValue;
                            mat_JVCdenseForSparse(i,j) = 0;
                            zeros_in_row++;
                        }
                    }
                    if( zeros_in_row == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( gen );
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(i, randomInt) = randomDouble;
                        mat_JVCdenseForSparse(i, randomInt) = randomDouble;
                    }
                }
                for( int j = 0; j < n; j++ ) {
                    int zeros_in_col = 0;
                    for( int i = 0; i < n; i++ ) {
                        if( SPML::Compare::IsZeroAbs( mat_JVCdenseForSparse(i, j) ) ) {
                            zeros_in_col++;
                        }
                    }
                    if( zeros_in_col == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( gen );
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(randomInt, j) = randomDouble;
                        mat_JVCdenseForSparse(randomInt, j) = randomDouble;
                    }
                }
            }
            // Матрицы размера 2N для учета неназначений
            arma::mat mat_JVCdense2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::ones );
            mat_JVCdense2N *= bigValue;
            arma::mat mat_JVCdenseForSparse2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::zeros );

            // Копируем основную часть матрицы в новую
            mat_JVCdense2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdense;
            mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdenseForSparse;
            //----------
            // FOR DENSE
            //old
//            arma::mat emptyDense = arma::mat( n, n, arma::fill::ones );
////            arma::mat emptyDense = arma::mat( n, n, arma::fill::randu );
//            emptyDense *= -halfBigValue;
            //new
            arma::mat emptyDense = arma::mat( n, n, arma::fill::ones );
            emptyDense *= bigValue;
            arma::vec emptyDenseDiag = arma::vec( n, arma::fill::ones );
            emptyDenseDiag *= halfBigValue;
            emptyDense.diag() = emptyDenseDiag;

//            arma::mat verbotenDense = arma::mat( n, n, arma::fill::ones );
////            arma::mat verbotenDense = arma::mat( n, n, arma::fill::randu );
//            verbotenDense *= bigValue;//halfBigValue;

            mat_JVCdense2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = emptyDense;
//            mat_JVCdense2N.submat( n, 0, ( 2 * n - 1 ), ( n - 1 ) ) = emptyDense;
//            mat_JVCdense2N.submat( n, n, ( 2 * n - 1 ), ( 2 * n - 1 ) ) = verbotenDense;
            mat_Mack2N = mat_JVCdense2N;
            mat_Hungarian2N = mat_JVCdense2N;
            //----------
            // FOR SPARSE
            arma::mat emptySparse = arma::mat( n, n, arma::fill::eye );
//            arma::mat emptySparse = arma::mat( n, n, arma::fill::zeros );
//            arma::vec emptySparseDiag = arma::vec( n, arma::fill::randu );
//            emptySparse.diag() = emptySparseDiag;
            emptySparse *= halfBigValue;

            arma::mat verbotenSparse = arma::mat( n, n, arma::fill::eye );
//            arma::mat verbotenSparse = arma::mat( n, n, arma::fill::zeros );
//            arma::vec verbotenSparseDiag = arma::vec( n, arma::fill::randu );
//            verbotenSparse.diag() = verbotenSparseDiag;
            verbotenSparse *= halfBigValue;// bigValue;

//            mat_JVCdenseForSparse2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = emptySparse;
//            mat_JVCdenseForSparse2N.submat( n, 0, ( 2 * n - 1 ), ( n - 1 ) ) = emptySparse;
            mat_JVCdenseForSparse2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = verbotenSparse;
            mat_JVCdenseForSparse2N.submat( n, 0, ( 2 * n - 1 ), ( n - 1 ) ) = verbotenSparse;
            mat_JVCdenseForSparse2N.submat( n, n, ( 2 * n - 1 ), ( 2 * n - 1 ) ) = verbotenSparse;

            mat_JVCdenseForSparse2N = mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( 2 * n - 1 ) );

            //----------
            // PRINT            
//            mat_JVCdense2N.print( "mat_JVCdense2N" );
//            mat_JVCdenseForSparse2N.print( "mat_JVCdenseForSparse2N" );
            //----------

            // Сделаем CSR матрицу для JVCsparse
            SPML::Sparse::MatrixDenseToCSR( mat_JVCdenseForSparse2N, mat_JVCsparse2N );

            // Запуск решений с замерами времени выполнения
            timer.at( "JVCdense" ).StartTimer();
            SPML::LAP::JVCdense( mat_JVCdense2N, 2*n, sp, infValue, resolution, actualJVCdense2N, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

            timer.at( "JVCsparse" ).StartTimer();
            int resSparse = SPML::LAP::JVCsparse( mat_JVCsparse2N.csr_val, mat_JVCsparse2N.csr_kk, mat_JVCsparse2N.csr_first,
                sp, infValue, resolution, actualJVCsparse2N, lapcostJVCsparse );
            timer.at( "JVCsparse" ).EndTimer();
            if( resSparse == 1 ) {
                assert( false );
            }

            timer.at( "Mack" ).StartTimer();
            SPML::LAP::Mack( mat_Mack2N, 2*n, sp, infValue, resolution, actualMack2N, lapcostMack );
            timer.at( "Mack" ).EndTimer();

            if( n <= boundN ) {
                timer.at( "Hungarian" ).StartTimer();
                SPML::LAP::Hungarian( mat_Hungarian2N, 2*n, sp, infValue, resolution, actualHungarian2N, lapcostHungarian );
                timer.at( "Hungarian" ).EndTimer();
            }

            // Проверим соответствие решений всех методов!
            std::vector< std::pair<int, int> > solutionJVCdense;
            std::vector< std::pair<int, int> > solutionJVCsparse;
            std::vector< std::pair<int, int> > solutionMack;
            std::vector< std::pair<int, int> > solutionHungarian;

            for( int i = 0; i < n; i++ ) {
                if( actualJVCdense2N(i) < n ) {
                    solutionJVCdense.push_back( std::make_pair( i, actualJVCdense2N(i) ) );
                }
                if( actualJVCsparse2N(i) < n ) {
                    solutionJVCsparse.push_back( std::make_pair( i, actualJVCsparse2N(i) ) );
                }
                if( actualMack2N(i) < n ) {
                    solutionMack.push_back( std::make_pair( i, actualMack2N(i) ) );
                }
                if( n <= boundN ) { // Для венгерки
                    if( actualHungarian2N(i) < n ) {
                        solutionHungarian.push_back( std::make_pair( i, actualHungarian2N(i) ) );
                    }
                }
            }

            if( ( solutionJVCdense.size() != solutionJVCsparse.size() ) ||
                ( solutionJVCdense.size() != solutionMack.size() ) ||
                ( ( n <= boundN ) && ( solutionJVCdense.size() != solutionHungarian.size() ) ) )
            {
                counter_assign++;
            } else {
                int size = solutionJVCdense.size();
                for( int i = 0; i < size; i++ ) {
                    if( ( solutionJVCdense[i].first != solutionJVCsparse[i].first ) ||
                        ( solutionJVCdense[i].second != solutionJVCsparse[i].second ) ||
                        ( solutionJVCdense[i].first != solutionMack[i].first ) ||
                        ( solutionJVCdense[i].second != solutionMack[i].second ) ||
                        ( ( n <= boundN ) && ( ( solutionJVCdense[i].first != solutionHungarian[i].first ) ||
                        ( solutionJVCdense[i].second != solutionHungarian[i].second ) ) ) )
                    {
                        counter_assign++;
                        break;
                    }
                }
            }

            // Оценим LAPCOST - суммарную стоимость назначения

            double totalcostJVCdense = 0.0;
            for( auto &elem : solutionJVCdense ) {
                totalcostJVCdense += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostJVCsparse = 0.0;
            for( auto &elem : solutionJVCsparse ) {
                totalcostJVCsparse += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostHungarian = 0.0;
            for( auto &elem : solutionHungarian ) {
                totalcostHungarian += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostMack = 0.0;
            for( auto &elem : solutionMack ) {
                totalcostMack+= mat_JVCdense( elem.first, elem.second );
            }

            bool eq_totalcostJVCdense_totalcostMack = std::abs( totalcostJVCdense - totalcostMack ) < 1e-5;
            bool eq_totalcostJVCdense_totalcostJVCsparse = std::abs( totalcostJVCdense - totalcostJVCsparse ) < 1e-5;
            bool eq_totalcostJVCdense_totalcostHungarian = false;
            if( n <= boundN ) {
                eq_totalcostJVCdense_totalcostHungarian = std::abs( totalcostJVCdense - totalcostHungarian ) < 1e-5;
            }

            if( !eq_totalcostJVCdense_totalcostMack ||
                !eq_totalcostJVCdense_totalcostJVCsparse ||
                ( ( n <= boundN ) && ( !eq_totalcostJVCdense_totalcostHungarian ) ) )
            {
                counter_lapcost++;
            }
        }
        if( print ) {
            for( auto &t : timer ) {
                std::cout << "timer " + t.first + " TimePerOp() = " << t.second.TimePerOp() << std::endl;
            }
        }

        for( auto &t : timer ) {
            if( ( n > boundN ) && ( t.first == "Hungarian" ) ) {
                continue;
            }
            timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() * 1.0e3 ); // мс
        }
        std::cout << "counter_lapcost=" << counter_lapcost << std::endl;
        std::cout << "counter_assign=" << counter_assign << std::endl;
        counter_lapcost_sum += counter_lapcost;
        counter_assign_sum += counter_assign;
    } // for n
    std::cout << "counter_lapcost_sum=" << counter_lapcost_sum << std::endl;
    std::cout << "counter_assign_sum=" << counter_assign_sum << std::endl;
#ifdef MATPLOTLIB
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }        
    for( auto &t : timeOfMethod ) {
        std::vector<double> dim = dimensionDouble.at( t.first );
        for( auto &ii : dim ) {
            ii *= 2;
        }
        plt::plot(
//            dimensionDouble.at( t.first ),
            dim,
            ( t.second ), // Y
            estimated_keywords.at( t.first )
            );
    }
    plt::grid( true );
    plt::xlim( dimensionXlim.front(), dimensionXlim.back() );
    plt::ylim( 0.0, 3.0 );
    plt::legend();
    plt::save( "sparse4to64.png", dpi );
    if( show ) {
        plt::show();
    }
    plt::close();
#endif
#ifdef PRINTTOTXT
    std::ofstream os;
    os.open( "sparse4to64.ods", std::ofstream::out );
    if( print ) {
        std::cout << "PRINTTOTXT..." << std::endl;
    }
    os << "N" << "\t";
    for( auto &t : timeOfMethod ) {
        os << t.first << "\t";
    }
    os << std::endl;
    for( int i = 0; i < dimensionXlim.size(); i++ ) {
        os << dimensionXlim[i] << "\t";
        for( auto &t : timeOfMethod ) {
            std::string method = t.first;
            double value = 0.0; //(t.second)[i];
            if( i < ( ( t.second ).size() ) ) {
                value = ( t.second )[i];
            }
            os << value << "\t";
        }
        os << std::endl;
    }
    os.close();
#endif
}

BOOST_AUTO_TEST_CASE( sparse5to50SeqExtr )
{
    SPML::LAP::TSearchParam sp = SPML::LAP::TSearchParam::SP_Max;
    double resolution = 1.0e-6;

    bool print = true; //false; //
    int cycle_count = 1000; //1;//
#ifdef MATPLOTLIB
    namespace plt = matplotlibcpp;
    double width = 100;
    double height = 50;

    const double in2mm = 25.4;// mm (fixed)
    const double dpi = 300;// dpi (variable)
    const double mm2px = dpi / in2mm;//
    size_t pixels_width = std::round( width * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
//    plt::title( "Время решения задачи о назначениях на разреженных матрицах" );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, [мс]" );
#endif

    std::vector<int> dimensionXlim = { 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 };
    std::vector<double> dimensionLong( dimensionXlim.begin(), dimensionXlim.end() );
    std::vector<double> dimensionShort( dimensionXlim.begin(), dimensionXlim.end() );

    int boundN = dimensionShort.back();

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "JVCdense", dimensionLong },
        { "JVCsparse", dimensionLong },
        { "SeqExtr", dimensionLong },
        { "SeqExtrCOO", dimensionLong },
        { "Mack", dimensionLong },
        { "Hungarian", dimensionShort }
    };

    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "JVCdense", {} },
        { "JVCsparse", {} },
        { "SeqExtr", {} },
        { "SeqExtrCOO", {} },
        { "Mack",{} },
        { "Hungarian",{} }
    };

    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
        { "SeqExtr", { { "color", "orange" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "SeqExtr" } } },
        { "SeqExtrCOO", { { "color", "yellow" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "SeqExtrCOO" } } },
        { "JVCdense", { { "color", "red" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCdense" } } },
        { "JVCsparse", { { "color", "magenta" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCsparse" } } },
        { "Mack", { { "color", "green" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "blue" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Hungarian" } } }
    };

    std::mt19937 generator; // Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_double_probability( 0.0, 1.0 ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности

    int counter_assign = 0;
    int counter_lapcost = 0;
    int counter_assign_sum = 0;
    int counter_lapcost_sum = 0;

    for( auto &n : dimensionXlim ) {
        counter_assign = 0;
        counter_lapcost = 0;

        std::uniform_int_distribution<int> random_uint_0_n( 0, ( n - 1 ) );

        SPML::Sparse::CMatrixCSR mat_JVCsparse;
        SPML::Sparse::CMatrixCSR mat_JVCsparse2N;
        SPML::Sparse::CMatrixCOO mat_COOsparse;

        arma::mat mat_JVCdense( n, n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse( n, n, arma::fill::zeros );
        arma::mat mat_Mack( n, n, arma::fill::zeros );
        arma::mat mat_Hungarian( n, n, arma::fill::zeros );

        arma::mat mat_JVCdense2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Mack2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Hungarian2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_SeqExtr2N( 2*n, 2*n, arma::fill::zeros );

        arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualJVCsparse = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );

        arma::ivec actualJVCdense2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualJVCsparse2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualMack2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualHungarian2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualSeqExtr2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualSeqExtrCOO2N = arma::ivec( 2*n, arma::fill::zeros );

        double lapcostJVCdense = 0.0, lapcostJVCsparse = 0.0, lapcostMack = 0.0, lapcostHungarian = 0.0,
            lapcostSeqExtr = 0.0, lapcostSeqExtrCOO = 0.0;

        std::map<std::string, SPML::Timing::CTimeKeeper> timer = {
            { "JVCdense", SPML::Timing::CTimeKeeper() },
            { "JVCsparse", SPML::Timing::CTimeKeeper() },
            { "SeqExtr", SPML::Timing::CTimeKeeper() },
            { "SeqExtrCOO", SPML::Timing::CTimeKeeper() },            
            { "Mack", SPML::Timing::CTimeKeeper() },
            { "Hungarian", SPML::Timing::CTimeKeeper() }
        };

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }
            arma::arma_rng::set_seed( cycle );
            mat_JVCdense.randu(); // 0..1
            mat_JVCdenseForSparse = mat_JVCdense;

//            bool doSparse = false;
            bool doSparse = true;

            double infValue = 1.0e7;
            double bigValue = 1.0e-6;
            double halfBigValue = 1.0e-5;

            if( doSparse ) {
                // Проредим матрицу в случайных местах
                double levelCut = 0.02; // Порог прореживания
                for( int i = 0; i < n; i++ ) {
                    int zeros_in_row = 0;
                    for( int j = 0; j < n; j++ ) {
                        double randomDouble = random_double_probability( generator ); // Случайное вещественное число от 0 до 1
                        if( randomDouble > levelCut ) {
                            mat_JVCdense(i, j) = -bigValue;//-infValue;
                            mat_JVCdenseForSparse(i,j) = 0;
                            zeros_in_row++;
                        }
                    }
                    if( zeros_in_row == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( generator );
                        double randomDouble = random_double_probability( generator ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(i, randomInt) = randomDouble;
                        mat_JVCdenseForSparse(i, randomInt) = randomDouble;
                    }
                }
                for( int j = 0; j < n; j++ ) {
                    int zeros_in_col = 0;
                    for( int i = 0; i < n; i++ ) {
                        if( SPML::Compare::IsZeroAbs( mat_JVCdenseForSparse(i, j) ) ) {
                            zeros_in_col++;
                        }
                    }
                    if( zeros_in_col == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( generator );
                        double randomDouble = random_double_probability( generator ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(randomInt, j) = randomDouble;
                        mat_JVCdenseForSparse(randomInt, j) = randomDouble;
                    }
                }
            }
            // Матрицы размера 2N для учета неназначений
            arma::mat mat_JVCdense2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::zeros );
            arma::mat mat_JVCdenseForSparse2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::zeros );

            // Копируем основную часть матрицы в новую
            mat_JVCdense2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdense;
            mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdenseForSparse;
            //----------
            // FOR DENSE
            arma::mat emptyDense = arma::mat( n, n, arma::fill::ones );
            emptyDense *= halfBigValue;
            arma::mat verbotenDense = arma::mat( n, n, arma::fill::ones );
            verbotenDense *= halfBigValue;

            mat_JVCdense2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = emptyDense;
            mat_JVCdense2N.submat( n, 0, ( 2 * n - 1 ), ( n - 1 ) ) = emptyDense;
            mat_JVCdense2N.submat( n, n, ( 2 * n - 1 ), ( 2 * n - 1 ) ) = verbotenDense;

            //----------
            // FOR SPARSE
            arma::mat emptySparse = arma::mat( n, n, arma::fill::eye );
            emptySparse *= halfBigValue;
            arma::mat verbotenSparse = arma::mat( n, n, arma::fill::eye );
            verbotenSparse *= bigValue;

            mat_JVCdenseForSparse2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = emptySparse;
            mat_JVCdenseForSparse2N.submat( n, 0, ( 2 * n - 1 ), ( n - 1 ) ) = emptySparse;
            mat_JVCdenseForSparse2N.submat( n, n, ( 2 * n - 1 ), ( 2 * n - 1 ) ) = verbotenSparse;

            mat_JVCdenseForSparse2N = mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( 2 * n - 1 ) );

            //----------
            // PRINT
//            mat_JVCdense.print( "mat_JVCdense" );
//            mat_JVCdense2N.print( "mat_JVCdense2N" );
//            mat_JVCdenseForSparse2N.print( "mat_JVCdenseForSparse2N" );
            //----------

            // Сделаем CSR матрицу для JVCsparse
            SPML::Sparse::MatrixDenseToCSR( mat_JVCdenseForSparse2N, mat_JVCsparse2N );
            mat_Mack2N = mat_JVCdense2N;
            mat_Hungarian2N = mat_JVCdense2N;
            mat_SeqExtr2N = mat_JVCdense2N;
            SPML::Sparse::MatrixDenseToCOO( mat_JVCdenseForSparse2N, mat_COOsparse );
//            for( int i = 0; i < n; i++ ) {
//                for( int j = 0; j < n; j++ ) {
//                    if( mat_JVCdense(i,j) > 0 ) {
//                        mat_COOsparse.coo_row.push_back( i );
//                        mat_COOsparse.coo_col.push_back( j );
//                        mat_COOsparse.coo_val.push_back( mat_JVCdense(i,j) );
//                    }
//                }
//            }

            // Запуск решений с замерами времени выполнения
            timer.at( "JVCdense" ).StartTimer();
            SPML::LAP::JVCdense( mat_JVCdense2N, 2*n, sp, infValue, resolution, actualJVCdense2N, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

            timer.at( "JVCsparse" ).StartTimer();
            int resSparse = SPML::LAP::JVCsparse( mat_JVCsparse2N.csr_val, mat_JVCsparse2N.csr_kk, mat_JVCsparse2N.csr_first,
                sp, infValue, resolution, actualJVCsparse2N, lapcostJVCsparse );
            timer.at( "JVCsparse" ).EndTimer();
            if( resSparse == 1 ) {
                assert( false );
            }

            timer.at( "SeqExtr" ).StartTimer();
            SPML::LAP::SequentalExtremum( mat_SeqExtr2N, sp, infValue, resolution, actualSeqExtr2N, lapcostSeqExtr );
            timer.at( "SeqExtr" ).EndTimer();

            timer.at( "SeqExtrCOO" ).StartTimer();
            SPML::LAP::SequentalExtremum( mat_COOsparse, sp, infValue, resolution, actualSeqExtrCOO2N, lapcostSeqExtrCOO );
            timer.at( "SeqExtrCOO" ).EndTimer();

            timer.at( "Mack" ).StartTimer();
            SPML::LAP::Mack( mat_Mack2N, 2*n, sp, infValue, resolution, actualMack2N, lapcostMack );
            timer.at( "Mack" ).EndTimer();

            if( n <= boundN ) {
                timer.at( "Hungarian" ).StartTimer();
                SPML::LAP::Hungarian( mat_Hungarian2N, 2*n, sp, infValue, resolution, actualHungarian2N, lapcostHungarian );
                timer.at( "Hungarian" ).EndTimer();
            }

            // Проверим соответствие решений всех методов!
            std::vector< std::pair<int, int> > solutionJVCdense;
            std::vector< std::pair<int, int> > solutionJVCsparse;
            std::vector< std::pair<int, int> > solutionMack;
            std::vector< std::pair<int, int> > solutionHungarian;
            std::vector< std::pair<int, int> > solutionSeqExtr;
            std::vector< std::pair<int, int> > solutionSeqExtrCOO;

            for( int i = 0; i < n; i++ ) {
                if( actualJVCdense2N(i) < n ) {
                    solutionJVCdense.push_back( std::make_pair( i, actualJVCdense2N(i) ) );
                }
                if( actualJVCsparse2N(i) < n ) {
                    solutionJVCsparse.push_back( std::make_pair( i, actualJVCsparse2N(i) ) );
                }
                if( actualMack2N(i) < n ) {
                    solutionMack.push_back( std::make_pair( i, actualMack2N(i) ) );
                }
                if( actualSeqExtr2N(i) < n ) {
                    solutionSeqExtr.push_back( std::make_pair( i, actualSeqExtr2N(i) ) );
                }
                if( actualSeqExtrCOO2N(i) < n ) {
                    solutionSeqExtrCOO.push_back( std::make_pair( i, actualSeqExtrCOO2N(i) ) );
                }
                if( n <= boundN ) { // Для венгерки
                    if( actualHungarian2N(i) < n ) {
                        solutionHungarian.push_back( std::make_pair( i, actualHungarian2N(i) ) );
                    }
                }
            }

            if( ( solutionJVCdense.size() != solutionJVCsparse.size() ) ||
                ( solutionJVCdense.size() != solutionMack.size() ) ||
                ( ( n <= boundN ) && ( solutionJVCdense.size() != solutionHungarian.size() ) ) )
            {
                counter_assign++;
            } else {
                int size = solutionJVCdense.size();
                for( int i = 0; i < size; i++ ) {
                    if( ( solutionJVCdense[i].first != solutionJVCsparse[i].first ) ||
                        ( solutionJVCdense[i].second != solutionJVCsparse[i].second ) ||
                        ( solutionJVCdense[i].first != solutionMack[i].first ) ||
                        ( solutionJVCdense[i].second != solutionMack[i].second ) ||
                        ( ( n <= boundN ) && ( ( solutionJVCdense[i].first != solutionHungarian[i].first ) ||
                        ( solutionJVCdense[i].second != solutionHungarian[i].second ) ) ) )
                    {
                        counter_assign++;
                        break;
                    }
                }
            }

            // Оценим LAPCOST - суммарную стоимость назначения

            double totalcostJVCdense = 0.0;
            for( auto &elem : solutionJVCdense ) {
                totalcostJVCdense += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostJVCsparse = 0.0;
            for( auto &elem : solutionJVCsparse ) {
                totalcostJVCsparse += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostHungarian = 0.0;
            for( auto &elem : solutionHungarian ) {
                totalcostHungarian += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostMack = 0.0;
            for( auto &elem : solutionMack ) {
                totalcostMack+= mat_JVCdense( elem.first, elem.second );
            }

            bool eq_totalcostJVCdense_totalcostMack = std::abs( totalcostJVCdense - totalcostMack ) < 1e-5;
            bool eq_totalcostJVCdense_totalcostJVCsparse = std::abs( totalcostJVCdense - totalcostJVCsparse ) < 1e-5;
            bool eq_totalcostJVCdense_totalcostHungarian = false;
            if( n <= boundN ) {
                eq_totalcostJVCdense_totalcostHungarian = std::abs( totalcostJVCdense - totalcostHungarian ) < 1e-5;
            }

            if( !eq_totalcostJVCdense_totalcostMack ||
                !eq_totalcostJVCdense_totalcostJVCsparse ||
                ( ( n <= boundN ) && ( !eq_totalcostJVCdense_totalcostHungarian ) ) )
            {
                counter_lapcost++;
            }
        }
        if( print ) {
            for( auto &t : timer ) {
                std::cout << "timer " + t.first + " TimePerOp() = " << t.second.TimePerOp() << std::endl;
            }
        }

        for( auto &t : timer ) {
            if( ( n > boundN ) && ( t.first == "Hungarian" ) ) {
                continue;
            }
            timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() * 1.0e3 ); // мс
        }
        std::cout << "counter_lapcost=" << counter_lapcost << std::endl;
        std::cout << "counter_assign=" << counter_assign << std::endl;
        counter_lapcost_sum += counter_lapcost;
        counter_assign_sum += counter_assign;
    } // for n
    std::cout << "counter_lapcost_sum=" << counter_lapcost_sum << std::endl;
    std::cout << "counter_assign_sum=" << counter_assign_sum << std::endl;
#ifdef MATPLOTLIB
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    for( auto &t : timeOfMethod ) {
        plt::plot(
            dimensionDouble.at( t.first ),
            ( t.second ), // Y
            estimated_keywords.at( t.first )
            );
    }
    plt::grid( true );
    plt::xlim( dimensionXlim.front(), dimensionXlim.back() );
    plt::legend();
    plt::save( "sparse5to50SeqExtr.png", dpi );
    if( show ) {
        plt::show();
    }
    plt::close();
#endif
#ifdef PRINTTOTXT
    std::ofstream os;
    os.open( "sparse5to50.ods", std::ofstream::out );
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    os << "N" << "\t";
    for( auto &t : timeOfMethod ) {
        os << t.first << "\t";
    }
    os << std::endl;
    for( int i = 0; i < dimensionXlim.size(); i++ ) {
        os << dimensionXlim[i] << "\t";
        for( auto &t : timeOfMethod ) {
            std::string method = t.first;
            double value = 0.0; //(t.second)[i];
            if( i < ( ( t.second ).size() ) ) {
                value = ( t.second )[i];
            }
            os << value << "\t";
        }
        os << std::endl;
    }
    os.close();
#endif
}

BOOST_AUTO_TEST_CASE( sparse100to1000 )
{
    SPML::LAP::TSearchParam sp = SPML::LAP::TSearchParam::SP_Max;
    double resolution = 1.0e-6;

    bool print = true; //false; //
    int cycle_count = 10;//1000; //

    std::mt19937 gen; ///< Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_0_1( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности
    std::uniform_real_distribution<double> random_double_probability( resolution, 1.0 - resolution ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности

#ifdef MATPLOTLIB
    namespace plt = matplotlibcpp;
    double width = 100;
    double height = 50;

    const double in2mm = 25.4;// mm (fixed)
    const double dpi = 300;// dpi (variable)
    const double mm2px = dpi / in2mm;//
    size_t pixels_width = std::round( width * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
//    plt::title( "Время решения задачи о назначениях на разреженных матрицах" );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, [мс]" );
#endif
    std::vector<int> dimensionXlim =     { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 };
    std::vector<double> dimensionLong =  { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 };

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "JVCdense", dimensionLong },
        { "JVCsparse", dimensionLong }
    };

    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "JVCdense", {} },
        { "JVCsparse", {} }
    };

    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
        { "JVCdense", { { "color", "red" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCdense" } } },
        { "JVCsparse", { { "color", "magenta" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCsparse" } } }
    };

    int counter_assign = 0;
    int counter_lapcost = 0;
    int counter_assign_sum = 0;
    int counter_lapcost_sum = 0;

    double scaleFactor = 1.0;

    for( auto &n : dimensionXlim ) {
        counter_assign = 0;
        counter_lapcost = 0;

        std::uniform_int_distribution<int> random_uint_0_n( 0, ( n - 1 ) );

        SPML::Sparse::CMatrixCSR mat_JVCsparse;
        SPML::Sparse::CMatrixCSR mat_JVCsparse2N;

        arma::mat mat_JVCdense( n, n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse( n, n, arma::fill::zeros );
        arma::mat mat_Mack( n, n, arma::fill::zeros );
        arma::mat mat_Hungarian( n, n, arma::fill::zeros );

        arma::mat mat_JVCdense2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Mack2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Hungarian2N( 2*n, 2*n, arma::fill::zeros );

        arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualJVCsparse = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );

        arma::ivec actualJVCdense2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualJVCsparse2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualMack2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualHungarian2N = arma::ivec( 2*n, arma::fill::zeros );

        double lapcostJVCdense = 0.0, lapcostJVCsparse = 0.0;//, lapcostMack = 0.0, lapcostHungarian = 0.0;

        std::map<std::string, SPML::Timing::CTimeKeeper> timer = {
            { "JVCdense", SPML::Timing::CTimeKeeper() },
            { "JVCsparse", SPML::Timing::CTimeKeeper() }//,
//            { "Mack", SPML::Timing::CTimeKeeper() },
//            { "Hungarian", SPML::Timing::CTimeKeeper() }
        };

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }            
            // old
//            arma::arma_rng::set_seed( cycle );
//            mat_JVCdense.randu();// 0..1

            // new
            gen.seed( cycle );
            for( int i = 0; i < n; i++ ) {
                for( int j = 0; j < n; j++ ) {
                    double randomNum = random_0_1( gen );
                    mat_JVCdense(i,j) = randomNum * scaleFactor;
                }
            }
            mat_JVCdenseForSparse = mat_JVCdense;

//            bool doSparse = false;
            bool doSparse = true;
            double infValue = 1.0e7;
            double bigValue = 1.0e-6;
            double halfBigValue = 1.0e-3;//infValue * 0.5;

            if( doSparse ) {
                // Проредим матрицу в случайных местах
                double levelCut = 0.02; // Порог прореживания
                for( int i = 0; i < n; i++ ) {
                    int zeros_in_row = 0;
                    for( int j = 0; j < n; j++ ) {
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        if( randomDouble > levelCut ) {
                            mat_JVCdense(i, j) = -infValue;
                            mat_JVCdenseForSparse(i,j) = 0;
                            zeros_in_row++;
                        }
                    }
                    if( zeros_in_row == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( gen );
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(i, randomInt) = randomDouble;
                        mat_JVCdenseForSparse(i, randomInt) = randomDouble;
                    }
                }
                for( int j = 0; j < n; j++ ) {
                    int zeros_in_col = 0;
                    for( int i = 0; i < n; i++ ) {
                        if( SPML::Compare::IsZeroAbs( mat_JVCdenseForSparse(i, j) ) ) {
                            zeros_in_col++;
                        }
                    }
                    if( zeros_in_col == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( gen );
                        double randomDouble = random_double_probability( gen ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(randomInt, j) = randomDouble;
                        mat_JVCdenseForSparse(randomInt, j) = randomDouble;
                    }
                }
            }
            // Матрицы размера 2N для учета неназначений
            arma::mat mat_JVCdense2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::zeros );
            arma::mat mat_JVCdenseForSparse2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::zeros );

            // Копируем основную часть матрицы в новую
            mat_JVCdense2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdense;
            mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdenseForSparse;
            //----------
            // FOR DENSE
            arma::mat emptyDense = arma::mat( n, n, arma::fill::ones );
            emptyDense *= halfBigValue;
            arma::mat verbotenDense = arma::mat( n, n, arma::fill::ones );
            verbotenDense *= halfBigValue;

            mat_JVCdense2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = emptyDense;
            mat_JVCdense2N.submat( n, 0, ( 2 * n - 1 ), ( n - 1 ) ) = emptyDense;
            mat_JVCdense2N.submat( n, n, ( 2 * n - 1 ), ( 2 * n - 1 ) ) = verbotenDense;

            //----------
            // FOR SPARSE
            arma::mat emptySparse = arma::mat( n, n, arma::fill::eye );
            emptySparse *= halfBigValue;
            arma::mat verbotenSparse = arma::mat( n, n, arma::fill::eye );
            verbotenSparse *= bigValue;

            mat_JVCdenseForSparse2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = emptySparse;
            mat_JVCdenseForSparse2N.submat( n, 0, ( 2 * n - 1 ), ( n - 1 ) ) = emptySparse;
            mat_JVCdenseForSparse2N.submat( n, n, ( 2 * n - 1 ), ( 2 * n - 1 ) ) = verbotenSparse;

            mat_JVCdenseForSparse2N = mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( 2 * n - 1 ) );

            //----------
            // PRINT
//            mat_JVCdense2N.print();
//            mat_JVCdenseForSparse2N.print();
            //----------

            // Сделаем CSR матрицу для JVCsparse
            SPML::Sparse::MatrixDenseToCSR( mat_JVCdenseForSparse2N, mat_JVCsparse2N );
            mat_Mack2N = mat_JVCdense2N;
            mat_Hungarian2N = mat_JVCdense2N;

            // Запуск решений с замерами времени выполнения
            timer.at( "JVCdense" ).StartTimer();
            SPML::LAP::JVCdense( mat_JVCdense2N, 2*n, sp, infValue, resolution, actualJVCdense2N, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

            timer.at( "JVCsparse" ).StartTimer();
            int resSparse = SPML::LAP::JVCsparse( mat_JVCsparse2N.csr_val, mat_JVCsparse2N.csr_kk, mat_JVCsparse2N.csr_first,
                sp, infValue, resolution, actualJVCsparse2N, lapcostJVCsparse );
            timer.at( "JVCsparse" ).EndTimer();
            if( resSparse == 1 ) {
                assert( false );
            }

            // Проверим соответствие решений всех методов!
            std::vector< std::pair<int, int> > solutionJVCdense;
            std::vector< std::pair<int, int> > solutionJVCsparse;
            std::vector< std::pair<int, int> > solutionMack;
            std::vector< std::pair<int, int> > solutionHungarian;

            for( int i = 0; i < n; i++ ) {
                if( actualJVCdense2N(i) < n ) {
                    solutionJVCdense.push_back( std::make_pair( i, actualJVCdense2N(i) ) );
                }
                if( actualJVCsparse2N(i) < n ) {
                    solutionJVCsparse.push_back( std::make_pair( i, actualJVCsparse2N(i) ) );
                }
            }

            if( ( solutionJVCdense.size() != solutionJVCsparse.size() ) )
            {
                counter_assign++;
            } else {
                int size = solutionJVCdense.size();
                for( int i = 0; i < size; i++ ) {
                    if( ( solutionJVCdense[i].first != solutionJVCsparse[i].first ) ||
                        ( solutionJVCdense[i].second != solutionJVCsparse[i].second ) )
                    {
                        counter_assign++;
                        break;
                    }
                }
            }

            // Оценим LAPCOST - суммарную стоимость назначения
            double totalcostJVCdense = 0.0;
            for( auto &elem : solutionJVCdense ) {
                totalcostJVCdense += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostJVCsparse = 0.0;
            for( auto &elem : solutionJVCsparse ) {
                totalcostJVCsparse += mat_JVCdense( elem.first, elem.second );
            }
            bool eq_totalcostJVCdense_totalcostJVCsparse = std::abs( totalcostJVCdense - totalcostJVCsparse ) < 1e-5;

            if( !eq_totalcostJVCdense_totalcostJVCsparse )
            {
                counter_lapcost++;
            }
        }
        if( print ) {
            for( auto &t : timer ) {
                std::cout << "timer " + t.first + " TimePerOp() = " << t.second.TimePerOp() << std::endl;
            }
        }

        for( auto &t : timer ) {
            timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() * 1.0e3 ); // мс
        }
        std::cout << "counter_lapcost=" << counter_lapcost << std::endl;
        std::cout << "counter_assign=" << counter_assign << std::endl;
        counter_lapcost_sum += counter_lapcost;
        counter_assign_sum += counter_assign;
    } // for n
    std::cout << "counter_lapcost_sum=" << counter_lapcost_sum << std::endl;
    std::cout << "counter_assign_sum=" << counter_assign_sum << std::endl;
#ifdef MATPLOTLIB
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    for( auto &t : timeOfMethod ) {
        plt::plot(
            dimensionDouble.at( t.first ),
            ( t.second ), // Y
            estimated_keywords.at( t.first )
            );
    }
    plt::grid( true );
    plt::xlim( dimensionXlim.front(), dimensionXlim.back() );
    plt::legend();
    plt::save( "sparse100to1000.png", dpi );
    if( show ) {
        plt::show();
    }
    plt::close();
#endif
#ifdef PRINTTOTXT
    std::ofstream os;
    os.open( "sparse100to1000.ods", std::ofstream::out );
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    os << "N" << "\t";
    for( auto &t : timeOfMethod ) {
        os << t.first << "\t";
    }
    os << std::endl;
    for( int i = 0; i < dimensionXlim.size(); i++ ) {
        os << dimensionXlim[i] << "\t";
        for( auto &t : timeOfMethod ) {
            std::string method = t.first;
            double value = 0.0; //(t.second)[i];
            if( i < ( ( t.second ).size() ) ) {
                value = ( t.second )[i];
            }
            os << value << "\t";
        }
        os << std::endl;
    }
    os.close();
#endif
}

BOOST_AUTO_TEST_CASE( sparse100to1000SeqExtr )
{
    SPML::LAP::TSearchParam sp = SPML::LAP::TSearchParam::SP_Max;
    double resolution = 1.0e-6;

    bool print = true; //false; //
    int cycle_count = 1;//1000; //
#ifdef MATPLOTLIB
    namespace plt = matplotlibcpp;
    double width = 100;
    double height = 50;

    const double in2mm = 25.4;// mm (fixed)
    const double dpi = 300;// dpi (variable)
    const double mm2px = dpi / in2mm;//
    size_t pixels_width = std::round( width * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
//    plt::title( "Время решения задачи о назначениях на разреженных матрицах" );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, [мс]" );
#endif
    std::vector<int> dimensionXlim =     { 100, 200, 300, 400, 500 }; //, 600, 700, 800, 900, 1000 };
    std::vector<double> dimensionLong =  { 100, 200, 300, 400, 500 }; //, 600, 700, 800, 900, 1000 };

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "JVCdense", dimensionLong },
        { "JVCsparse", dimensionLong },
//        { "SeqExtr", dimensionLong },
        { "SeqExtrCOO", dimensionLong }
    };

    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "JVCdense", {} },
        { "JVCsparse", {} },
//        { "SeqExtr", {} },
        { "SeqExtrCOO", {} }
    };

    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
//        { "SeqExtr", { { "color", "orange" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "SeqExtr" } } },
        { "SeqExtrCOO", { { "color", "yellow" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "SeqExtrCOO" } } },
        { "JVCdense", { { "color", "red" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCdense" } } },
        { "JVCsparse", { { "color", "magenta" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCsparse" } } }
    };

    std::mt19937 generator; // Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_double_probability( 0.0, 1.0 ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности

    int counter_assign = 0;
    int counter_lapcost = 0;
    int counter_assign_sum = 0;
    int counter_lapcost_sum = 0;

    for( auto &n : dimensionXlim ) {
        counter_assign = 0;
        counter_lapcost = 0;

        std::uniform_int_distribution<int> random_uint_0_n( 0, ( n - 1 ) );

        SPML::Sparse::CMatrixCSR mat_JVCsparse;
        SPML::Sparse::CMatrixCSR mat_JVCsparse2N;
        SPML::Sparse::CMatrixCOO mat_COOsparse;

        arma::mat mat_JVCdense( n, n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse( n, n, arma::fill::zeros );
        arma::mat mat_Mack( n, n, arma::fill::zeros );
        arma::mat mat_Hungarian( n, n, arma::fill::zeros );

        arma::mat mat_JVCdense2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Mack2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Hungarian2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_SeqExtr2N( 2*n, 2*n, arma::fill::zeros );

        arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualJVCsparse = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );

        arma::ivec actualJVCdense2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualJVCsparse2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualMack2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualHungarian2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualSeqExtr2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualSeqExtrCOO2N = arma::ivec( 2*n, arma::fill::zeros );

        double lapcostJVCdense = 0.0, lapcostJVCsparse = 0.0, lapcostMack = 0.0, lapcostHungarian = 0.0,
            lapcostSeqExtr = 0.0, lapcostSeqExtrCOO = 0.0;

        ///

        std::map<std::string, SPML::Timing::CTimeKeeper> timer = {
            { "JVCdense", SPML::Timing::CTimeKeeper() },
            { "JVCsparse", SPML::Timing::CTimeKeeper() },
//            { "SeqExtr", SPML::Timing::CTimeKeeper() },
            { "SeqExtrCOO", SPML::Timing::CTimeKeeper() }
        };

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }
            arma::arma_rng::set_seed( cycle );
            mat_JVCdense.randu(); // 0..1
            mat_JVCdenseForSparse = mat_JVCdense;

//            bool doSparse = false;
            bool doSparse = true;
            double infValue = 1.0e7;
            double bigValue = 1.0e-6;
            double halfBigValue = 1.0e-3;//infValue * 0.5;

            if( doSparse ) {
                // Проредим матрицу в случайных местах
                double levelCut = 0.02; // Порог прореживания
                for( int i = 0; i < n; i++ ) {
                    int zeros_in_row = 0;
                    for( int j = 0; j < n; j++ ) {
                        double randomDouble = random_double_probability( generator ); // Случайное вещественное число от 0 до 1
                        if( randomDouble > levelCut ) {
                            mat_JVCdense(i, j) = -infValue;
                            mat_JVCdenseForSparse(i,j) = 0;
                            zeros_in_row++;
                        }
                    }
                    if( zeros_in_row == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( generator );
                        double randomDouble = random_double_probability( generator ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(i, randomInt) = randomDouble;
                        mat_JVCdenseForSparse(i, randomInt) = randomDouble;
                    }
                }
                for( int j = 0; j < n; j++ ) {
                    int zeros_in_col = 0;
                    for( int i = 0; i < n; i++ ) {
                        if( SPML::Compare::IsZeroAbs( mat_JVCdenseForSparse(i, j) ) ) {
                            zeros_in_col++;
                        }
                    }
                    if( zeros_in_col == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( generator );
                        double randomDouble = random_double_probability( generator ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(randomInt, j) = randomDouble;
                        mat_JVCdenseForSparse(randomInt, j) = randomDouble;
                    }
                }
            }
            // Матрицы размера 2N для учета неназначений
            arma::mat mat_JVCdense2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::zeros );
            arma::mat mat_JVCdenseForSparse2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::zeros );

            // Копируем основную часть матрицы в новую
            mat_JVCdense2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdense;
            mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdenseForSparse;
            //----------
            // FOR DENSE
            arma::mat emptyDense = arma::mat( n, n, arma::fill::ones );
            emptyDense *= halfBigValue;
            arma::mat verbotenDense = arma::mat( n, n, arma::fill::ones );
            verbotenDense *= halfBigValue;

            mat_JVCdense2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = emptyDense;
            mat_JVCdense2N.submat( n, 0, ( 2 * n - 1 ), ( n - 1 ) ) = emptyDense;
            mat_JVCdense2N.submat( n, n, ( 2 * n - 1 ), ( 2 * n - 1 ) ) = verbotenDense;

            //----------
            // FOR SPARSE
            arma::mat emptySparse = arma::mat( n, n, arma::fill::eye );
            emptySparse *= halfBigValue;
            arma::mat verbotenSparse = arma::mat( n, n, arma::fill::eye );
            verbotenSparse *= bigValue;

            mat_JVCdenseForSparse2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = emptySparse;
            mat_JVCdenseForSparse2N.submat( n, 0, ( 2 * n - 1 ), ( n - 1 ) ) = emptySparse;
            mat_JVCdenseForSparse2N.submat( n, n, ( 2 * n - 1 ), ( 2 * n - 1 ) ) = verbotenSparse;

            mat_JVCdenseForSparse2N = mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( 2 * n - 1 ) );

            //----------
            // PRINT
//            mat_JVCdense2N.print();
//            mat_JVCdenseForSparse2N.print();
            //----------

            // Сделаем CSR матрицу для JVCsparse
            SPML::Sparse::MatrixDenseToCSR( mat_JVCdenseForSparse2N, mat_JVCsparse2N );
            mat_Mack2N = mat_JVCdense2N;
            mat_Hungarian2N = mat_JVCdense2N;
            mat_SeqExtr2N = mat_JVCdense2N;
            SPML::Sparse::MatrixDenseToCOO( mat_JVCdenseForSparse2N, mat_COOsparse );

            // Запуск решений с замерами времени выполнения
            timer.at( "JVCdense" ).StartTimer();
            SPML::LAP::JVCdense( mat_JVCdense2N, 2*n, sp, infValue, resolution, actualJVCdense2N, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

//            timer.at( "SeqExtr" ).StartTimer();
//            SPML::LAP::SequentalExtremum( mat_SeqExtr2N, sp, infValue, resolution, actualSeqExtr2N, lapcostSeqExtr );
//            timer.at( "SeqExtr" ).EndTimer();

            timer.at( "SeqExtrCOO" ).StartTimer();
            SPML::LAP::SequentalExtremum( mat_COOsparse, sp, infValue, resolution, actualSeqExtrCOO2N, lapcostSeqExtrCOO );
            timer.at( "SeqExtrCOO" ).EndTimer();

            timer.at( "JVCsparse" ).StartTimer();
            int resSparse = SPML::LAP::JVCsparse( mat_JVCsparse2N.csr_val, mat_JVCsparse2N.csr_kk, mat_JVCsparse2N.csr_first,
                sp, infValue, resolution, actualJVCsparse2N, lapcostJVCsparse );
            timer.at( "JVCsparse" ).EndTimer();
            if( resSparse == 1 ) {
                assert( false );
            }

            // Проверим соответствие решений всех методов!
            std::vector< std::pair<int, int> > solutionJVCdense;
            std::vector< std::pair<int, int> > solutionJVCsparse;
//            std::vector< std::pair<int, int> > solutionSeqExtr;
            std::vector< std::pair<int, int> > solutionSeqExtrCOO;

            for( int i = 0; i < n; i++ ) {
                if( actualJVCdense2N(i) < n ) {
                    solutionJVCdense.push_back( std::make_pair( i, actualJVCdense2N(i) ) );
                }
                if( actualJVCsparse2N(i) < n ) {
                    solutionJVCsparse.push_back( std::make_pair( i, actualJVCsparse2N(i) ) );
                }
            }

            if( ( solutionJVCdense.size() != solutionJVCsparse.size() ) )
            {
                counter_assign++;
            } else {
                int size = solutionJVCdense.size();
                for( int i = 0; i < size; i++ ) {
                    if( ( solutionJVCdense[i].first != solutionJVCsparse[i].first ) ||
                        ( solutionJVCdense[i].second != solutionJVCsparse[i].second ) )
                    {
                        counter_assign++;
                        break;
                    }
                }
            }

            // Оценим LAPCOST - суммарную стоимость назначения
            double totalcostJVCdense = 0.0;
            for( auto &elem : solutionJVCdense ) {
                totalcostJVCdense += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostJVCsparse = 0.0;
            for( auto &elem : solutionJVCsparse ) {
                totalcostJVCsparse += mat_JVCdense( elem.first, elem.second );
            }
            bool eq_totalcostJVCdense_totalcostJVCsparse = std::abs( totalcostJVCdense - totalcostJVCsparse ) < 1e-5;

            if( !eq_totalcostJVCdense_totalcostJVCsparse )
            {
                counter_lapcost++;
            }
        }
        if( print ) {
            for( auto &t : timer ) {
                std::cout << "timer " + t.first + " TimePerOp() = " << t.second.TimePerOp() << std::endl;
            }
        }

        for( auto &t : timer ) {
            timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() * 1.0e3 ); // мс
        }
        std::cout << "counter_lapcost=" << counter_lapcost << std::endl;
        std::cout << "counter_assign=" << counter_assign << std::endl;
        counter_lapcost_sum += counter_lapcost;
        counter_assign_sum += counter_assign;
    } // for n
    std::cout << "counter_lapcost_sum=" << counter_lapcost_sum << std::endl;
    std::cout << "counter_assign_sum=" << counter_assign_sum << std::endl;
#ifdef MATPLOTLIB
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    for( auto &t : timeOfMethod ) {
        plt::plot(
            dimensionDouble.at( t.first ),
            ( t.second ), // Y
            estimated_keywords.at( t.first )
            );
    }
    plt::grid( true );
    plt::xlim( dimensionXlim.front(), dimensionXlim.back() );
    plt::legend();
    plt::save( "sparse100to1000SeqExtr.png", dpi );
    if( show ) {
        plt::show();
    }
    plt::close();
#endif
#ifdef PRINTTOTXT
    std::ofstream os;
    os.open( "sparse100to1000.ods", std::ofstream::out );
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    os << "N" << "\t";
    for( auto &t : timeOfMethod ) {
        os << t.first << "\t";
    }
    os << std::endl;
    for( int i = 0; i < dimensionXlim.size(); i++ ) {
        os << dimensionXlim[i] << "\t";
        for( auto &t : timeOfMethod ) {
            std::string method = t.first;
            double value = 0.0; //(t.second)[i];
            if( i < ( ( t.second ).size() ) ) {
                value = ( t.second )[i];
            }
            os << value << "\t";
        }
        os << std::endl;
    }
    os.close();
#endif
}

BOOST_AUTO_TEST_CASE( sparse500to5000 )
{
    SPML::LAP::TSearchParam sp = SPML::LAP::TSearchParam::SP_Max;
    double resolution = 1.0e-6;
    bool print = true; //false; //
    int cycle_count = 10;//1000; //
#ifdef MATPLOTLIB
    namespace plt = matplotlibcpp;
    double width = 100;
    double height = 50;

    const double in2mm = 25.4;// mm (fixed)
    const double dpi = 300;// dpi (variable)
    const double mm2px = dpi / in2mm;//
    size_t pixels_width = std::round( width * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
//    plt::title( "Время решения задачи о назначениях на разреженных матрицах" );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, [мс]" );
#endif
    std::vector<int> dimensionXlim;// =     { 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 };
    std::vector<double> dimensionLong;// =  { 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 };
    for( int i = 500; i <= 5000; i = i + 500 ) {
        dimensionXlim.push_back( i );
        dimensionLong.push_back( static_cast<double>( i ) );
    }

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "JVCsparse", dimensionLong }//,
    };

    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "JVCsparse", {} }//,
    };

    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
        { "JVCsparse", { { "color", "magenta" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCsparse" } } }//,
    };

    std::mt19937 generator; // Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_double_probability( 0.0, 1.0 ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности

    int counter_assign = 0;
    int counter_lapcost = 0;
    int counter_assign_sum = 0;
    int counter_lapcost_sum = 0;

    for( auto &n : dimensionXlim ) {
        counter_assign = 0;
        counter_lapcost = 0;

        std::uniform_int_distribution<int> random_uint_0_n( 0, ( n - 1 ) );

        SPML::Sparse::CMatrixCSR mat_JVCsparse;
        SPML::Sparse::CMatrixCSR mat_JVCsparse2N;

        arma::mat mat_JVCdense( n, n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse( n, n, arma::fill::zeros );
        arma::mat mat_Mack( n, n, arma::fill::zeros );
        arma::mat mat_Hungarian( n, n, arma::fill::zeros );

        arma::mat mat_JVCdense2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Mack2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Hungarian2N( 2*n, 2*n, arma::fill::zeros );

        arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualJVCsparse = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );

        arma::ivec actualJVCdense2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualJVCsparse2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualMack2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualHungarian2N = arma::ivec( 2*n, arma::fill::zeros );

        double lapcostJVCsparse = 0.0;

        std::map<std::string, SPML::Timing::CTimeKeeper> timer = {
            { "JVCsparse", SPML::Timing::CTimeKeeper() }//,
        };

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }
            arma::arma_rng::set_seed( cycle );
            mat_JVCdense.randu(); // 0..1
            mat_JVCdenseForSparse = mat_JVCdense;

//            bool doSparse = false;
            bool doSparse = true;

            double infValue = 1.0e7;
            double bigValue = 1.0e-6;
            double halfBigValue = 1.0e-3;

            if( doSparse ) {
                // Проредим матрицу в случайных местах
                double levelCut = 0.2; // Порог прореживания
                for( int i = 0; i < n; i++ ) {
                    int zeros_in_row = 0;
                    for( int j = 0; j < n; j++ ) {
                        double randomDouble = random_double_probability( generator ); // Случайное вещественное число от 0 до 1
                        if( randomDouble > levelCut ) {
                            mat_JVCdense(i, j) = -infValue;
                            mat_JVCdenseForSparse(i,j) = 0;
                            zeros_in_row++;
                        }
                    }
                    if( zeros_in_row == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( generator );
                        double randomDouble = random_double_probability( generator ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(i, randomInt) = randomDouble;
                        mat_JVCdenseForSparse(i, randomInt) = randomDouble;
                    }
                }
                for( int j = 0; j < n; j++ ) {
                    int zeros_in_col = 0;
                    for( int i = 0; i < n; i++ ) {
                        if( SPML::Compare::IsZeroAbs( mat_JVCdenseForSparse(i, j) ) ) {
                            zeros_in_col++;
                        }
                    }
                    if( zeros_in_col == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( generator );
                        double randomDouble = random_double_probability( generator ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(randomInt, j) = randomDouble;
                        mat_JVCdenseForSparse(randomInt, j) = randomDouble;
                    }
                }
            }
            // Матрицы размера 2N для учета неназначений
            arma::mat mat_JVCdense2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::zeros );
            arma::mat mat_JVCdenseForSparse2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::zeros );

            // Копируем основную часть матрицы в новую
            mat_JVCdense2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdense;
            mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdenseForSparse;
            //----------
            // FOR DENSE
            arma::mat emptyDense = arma::mat( n, n, arma::fill::ones );
            emptyDense *= halfBigValue;
            arma::mat verbotenDense = arma::mat( n, n, arma::fill::ones );
            verbotenDense *= halfBigValue;

            mat_JVCdense2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = emptyDense;
            mat_JVCdense2N.submat( n, 0, ( 2 * n - 1 ), ( n - 1 ) ) = emptyDense;
            mat_JVCdense2N.submat( n, n, ( 2 * n - 1 ), ( 2 * n - 1 ) ) = verbotenDense;

            //----------
            // FOR SPARSE
            arma::mat emptySparse = arma::mat( n, n, arma::fill::eye );
            emptySparse *= halfBigValue;

            arma::mat verbotenSparse = arma::mat( n, n, arma::fill::eye );
            verbotenSparse *= bigValue;//-infValue;//-halfInfValue;//

            mat_JVCdenseForSparse2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = emptySparse;
            mat_JVCdenseForSparse2N.submat( n, 0, ( 2 * n - 1 ), ( n - 1 ) ) = emptySparse;
            mat_JVCdenseForSparse2N.submat( n, n, ( 2 * n - 1 ), ( 2 * n - 1 ) ) = verbotenSparse;//infMat;

            mat_JVCdenseForSparse2N = mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( 2 * n - 1 ) );

            //----------
            // PRINT
//            mat_JVCdense2N.print();
//            mat_JVCdenseForSparse2N.print();
            //----------

            // Сделаем CSR матрицу для JVCsparse
            SPML::Sparse::MatrixDenseToCSR( mat_JVCdenseForSparse2N, mat_JVCsparse2N );
            mat_Mack2N = mat_JVCdense2N;
            mat_Hungarian2N = mat_JVCdense2N;

            timer.at( "JVCsparse" ).StartTimer();
            int resSparse = SPML::LAP::JVCsparse( mat_JVCsparse2N.csr_val, mat_JVCsparse2N.csr_kk, mat_JVCsparse2N.csr_first,
                sp, infValue, resolution, actualJVCsparse2N, lapcostJVCsparse );
            timer.at( "JVCsparse" ).EndTimer();
            if( resSparse == 1 ) {
                assert( false );
            }

            // Проверим соответствие решений всех методов!
            std::vector< std::pair<int, int> > solutionJVCdense;
            std::vector< std::pair<int, int> > solutionJVCsparse;
            std::vector< std::pair<int, int> > solutionMack;
            std::vector< std::pair<int, int> > solutionHungarian;

            for( int i = 0; i < n; i++ ) {
                if( actualJVCsparse2N(i) < n ) {
                    solutionJVCsparse.push_back( std::make_pair( i, actualJVCsparse2N(i) ) );
                }
            }
        }
        if( print ) {
            for( auto &t : timer ) {
                std::cout << "timer " + t.first + " TimePerOp() = " << t.second.TimePerOp() << std::endl;
            }
        }

        for( auto &t : timer ) {
            timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() * 1.0e3 ); // мс
        }
        std::cout << "counter_lapcost=" << counter_lapcost << std::endl;
        std::cout << "counter_assign=" << counter_assign << std::endl;
        counter_lapcost_sum += counter_lapcost;
        counter_assign_sum += counter_assign;
    } // for n
    std::cout << "counter_lapcost_sum=" << counter_lapcost_sum << std::endl;
    std::cout << "counter_assign_sum=" << counter_assign_sum << std::endl;
#ifdef MATPLOTLIB
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    for( auto &t : timeOfMethod ) {
        plt::plot(
            dimensionDouble.at( t.first ),
            ( t.second ), // Y
            estimated_keywords.at( t.first )
            );
    }
    plt::grid( true );
    plt::xlim( dimensionXlim.front(), dimensionXlim.back() );
    plt::legend();
    plt::save( "sparse500to5000.png", dpi );
    if( show ) {
        plt::show();
    }
    plt::close();
#endif
#ifdef PRINTTOTXT
    std::ofstream os;
    os.open( "sparse500to5000.ods", std::ofstream::out );
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    os << "N" << "\t";
    for( auto &t : timeOfMethod ) {
        os << t.first << "\t";
    }
    os << std::endl;
    for( int i = 0; i < dimensionXlim.size(); i++ ) {
        os << dimensionXlim[i] << "\t";
        for( auto &t : timeOfMethod ) {
            std::string method = t.first;
            double value = 0.0; //(t.second)[i];
            if( i < ( ( t.second ).size() ) ) {
                value = ( t.second )[i];
            }
            os << value << "\t";
        }
        os << std::endl;
    }
    os.close();
#endif
}

BOOST_AUTO_TEST_CASE( time_table )
{
    SPML::LAP::TSearchParam sp = SPML::LAP::TSearchParam::SP_Max;
    double resolution = 1.0e-6;

    bool print = true; //false; //
    int cycle_count = 10; //1;//
#ifdef MATPLOTLIB
    namespace plt = matplotlibcpp;
    double width = 100;
    double height = 50;

    const double in2mm = 25.4;// mm (fixed)
    const double dpi = 300;// dpi (variable)
    const double mm2px = dpi / in2mm;//
    size_t pixels_width = std::round( width * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
//    plt::title( "Время решения задачи о назначениях на разреженных матрицах" );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, [мс]" );
#endif
    std::vector<int> dimensionXlim    = { 5, 10, 25, 50, 100, 150, 200, 250, 500, 1000 };
    std::vector<double> dimensionLong = { 5, 10, 25, 50, 100, 150, 200, 250, 500, 1000 };
    std::vector<double> dimensionShort= { 5, 10, 25, 50, 100, 150, 200 };
//    std::vector<int> dimensionXlim    = { 5, 10, 25, 50 };
//    std::vector<double> dimensionLong = { 5, 10, 25, 50 };
//    std::vector<double> dimensionShort= { 5, 10, 25, 50 };

    int boundN = dimensionShort.back();

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "JVCdense", dimensionLong },
        { "JVCsparse", dimensionLong },
        { "Mack", dimensionShort },
        { "Hungarian", dimensionShort },
        { "SeqExtr", dimensionShort },
        { "SeqExtrCOO", dimensionShort }
    };

    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "JVCdense", {} },
        { "JVCsparse", {} },
        { "Mack", {} },
        { "Hungarian", {} },
        { "SeqExtr", {} },
        { "SeqExtrCOO", {} }
    };

    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
        { "JVCdense", { { "color", "red" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCdense" } } },
        { "JVCsparse", { { "color", "magenta" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVCsparse" } } },
        { "Mack", { { "color", "green" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "blue" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Hungarian" } } },
        { "SeqExtr", { { "color", "orange" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "SeqExtr" } } },
        { "SeqExtrCOO", { { "color", "yellow" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "SeqExtrCOO" } } }
    };

    std::mt19937 generator; // Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_double_probability( 0.0, 1.0 ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности

    int counter_assign = 0;
    int counter_lapcost = 0;
    int counter_assign_sum = 0;
    int counter_lapcost_sum = 0;

    for( auto &n : dimensionXlim ) {
        counter_assign = 0;
        counter_lapcost = 0;

        std::uniform_int_distribution<int> random_uint_0_n( 0, ( n - 1 ) );

        SPML::Sparse::CMatrixCSR mat_JVCsparse;
        SPML::Sparse::CMatrixCSR mat_JVCsparse2N;
        SPML::Sparse::CMatrixCOO mat_COOsparse;

        arma::mat mat_JVCdense( n, n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse( n, n, arma::fill::zeros );
        arma::mat mat_Mack( n, n, arma::fill::zeros );
        arma::mat mat_Hungarian( n, n, arma::fill::zeros );

        arma::mat mat_JVCdense2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Mack2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_Hungarian2N( 2*n, 2*n, arma::fill::zeros );
        arma::mat mat_SeqExtr2N( 2*n, 2*n, arma::fill::zeros );

        arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualJVCsparse = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );

        arma::ivec actualJVCdense2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualJVCsparse2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualMack2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualHungarian2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualSeqExtr2N = arma::ivec( 2*n, arma::fill::zeros );
        arma::ivec actualSeqExtrCOO2N = arma::ivec( 2*n, arma::fill::zeros );

        double lapcostJVCdense = 0.0, lapcostJVCsparse = 0.0, lapcostMack = 0.0, lapcostHungarian = 0.0,
            lapcostSeqExtr = 0.0, lapcostSeqExtrCOO = 0.0;

        std::map<std::string, SPML::Timing::CTimeKeeper> timer = {
            { "JVCdense", SPML::Timing::CTimeKeeper() },
            { "JVCsparse", SPML::Timing::CTimeKeeper() },
            { "Mack", SPML::Timing::CTimeKeeper() },
            { "Hungarian", SPML::Timing::CTimeKeeper() },
            { "SeqExtr", SPML::Timing::CTimeKeeper() },
            { "SeqExtrCOO", SPML::Timing::CTimeKeeper() }
        };

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }
            arma::arma_rng::set_seed( cycle );
            mat_JVCdense.randu(); // 0..1
            mat_JVCdenseForSparse = mat_JVCdense;

//            bool doSparse = false;
            bool doSparse = true;

            double infValue = 1.0e7;
            double bigValue = 1.0e-6;
            double halfBigValue = 1.0e-3;

            if( doSparse ) {
                // Проредим матрицу в случайных местах
                double levelCut = 0.02; // Порог прореживания
                for( int i = 0; i < n; i++ ) {
                    int zeros_in_row = 0;
                    for( int j = 0; j < n; j++ ) {
                        double randomDouble = random_double_probability( generator ); // Случайное вещественное число от 0 до 1
                        if( randomDouble > levelCut ) {
                            mat_JVCdense(i, j) = -infValue;
                            mat_JVCdenseForSparse(i,j) = 0;
                            zeros_in_row++;
                        }
                    }
                    if( zeros_in_row == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( generator );
                        double randomDouble = random_double_probability( generator ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(i, randomInt) = randomDouble;
                        mat_JVCdenseForSparse(i, randomInt) = randomDouble;
                    }
                }
                for( int j = 0; j < n; j++ ) {
                    int zeros_in_col = 0;
                    for( int i = 0; i < n; i++ ) {
                        if( SPML::Compare::IsZeroAbs( mat_JVCdenseForSparse(i, j) ) ) {
                            zeros_in_col++;
                        }
                    }
                    if( zeros_in_col == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( generator );
                        double randomDouble = random_double_probability( generator ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(randomInt, j) = randomDouble;
                        mat_JVCdenseForSparse(randomInt, j) = randomDouble;
                    }
                }
            }
            // Матрицы размера 2N для учета неназначений
            arma::mat mat_JVCdense2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::zeros );
            arma::mat mat_JVCdenseForSparse2N = arma::mat( ( n * 2 ), ( n * 2 ), arma::fill::zeros );

            // Копируем основную часть матрицы в новую
            mat_JVCdense2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdense;
            mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( n - 1 ) ) = mat_JVCdenseForSparse;
            //----------
            // FOR DENSE
            arma::mat emptyDense = arma::mat( n, n, arma::fill::ones );
            emptyDense *= halfBigValue;
            arma::mat verbotenDense = arma::mat( n, n, arma::fill::ones );
            verbotenDense *= halfBigValue;

            mat_JVCdense2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = emptyDense;
            mat_JVCdense2N.submat( n, 0, ( 2 * n - 1 ), ( n - 1 ) ) = emptyDense;
            mat_JVCdense2N.submat( n, n, ( 2 * n - 1 ), ( 2 * n - 1 ) ) = verbotenDense;

            //----------
            // FOR SPARSE
            arma::mat emptySparse = arma::mat( n, n, arma::fill::eye );
            emptySparse *= halfBigValue;
            arma::mat verbotenSparse = arma::mat( n, n, arma::fill::eye );
            verbotenSparse *= bigValue;

            mat_JVCdenseForSparse2N.submat( 0, n, ( n - 1 ), ( 2 * n - 1 ) ) = emptySparse;
            mat_JVCdenseForSparse2N.submat( n, 0, ( 2 * n - 1 ), ( n - 1 ) ) = emptySparse;
            mat_JVCdenseForSparse2N.submat( n, n, ( 2 * n - 1 ), ( 2 * n - 1 ) ) = verbotenSparse;

            mat_JVCdenseForSparse2N = mat_JVCdenseForSparse2N.submat( 0, 0, ( n - 1 ), ( 2 * n - 1 ) );

            //----------
            // PRINT
//            mat_JVCdense2N.print( "mat_JVCdense2N" );
//            mat_JVCdenseForSparse2N.print( "mat_JVCdenseForSparse2N" );
            //----------

            // Сделаем CSR матрицу для JVCsparse
            SPML::Sparse::MatrixDenseToCSR( mat_JVCdenseForSparse2N, mat_JVCsparse2N );
            mat_Mack2N = mat_JVCdense2N;
            mat_Hungarian2N = mat_JVCdense2N;
            mat_SeqExtr2N = mat_JVCdense2N;
            SPML::Sparse::MatrixDenseToCOO( mat_JVCdenseForSparse2N, mat_COOsparse );

            // Запуск решений с замерами времени выполнения
            timer.at( "JVCdense" ).StartTimer();
            SPML::LAP::JVCdense( mat_JVCdense2N, 2*n, sp, infValue, resolution, actualJVCdense2N, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

            timer.at( "JVCsparse" ).StartTimer();
            int resSparse = SPML::LAP::JVCsparse( mat_JVCsparse2N.csr_val, mat_JVCsparse2N.csr_kk, mat_JVCsparse2N.csr_first,
                sp, infValue, resolution, actualJVCsparse2N, lapcostJVCsparse );
            timer.at( "JVCsparse" ).EndTimer();
            if( resSparse == 1 ) {
                assert( false );
            }

            if( n <= boundN ) {
                timer.at( "Mack" ).StartTimer();
                SPML::LAP::Mack( mat_Mack2N, 2*n, sp, infValue, resolution, actualMack2N, lapcostMack );
                timer.at( "Mack" ).EndTimer();

                timer.at( "Hungarian" ).StartTimer();
                SPML::LAP::Hungarian( mat_Hungarian2N, 2*n, sp, infValue, resolution, actualHungarian2N, lapcostHungarian );
                timer.at( "Hungarian" ).EndTimer();

                timer.at( "SeqExtr" ).StartTimer();
                SPML::LAP::SequentalExtremum( mat_SeqExtr2N, sp, infValue, resolution, actualSeqExtr2N, lapcostSeqExtr );
                timer.at( "SeqExtr" ).EndTimer();

                timer.at( "SeqExtrCOO" ).StartTimer();
                SPML::LAP::SequentalExtremum( mat_COOsparse, sp, infValue, resolution, actualSeqExtrCOO2N, lapcostSeqExtrCOO );
                timer.at( "SeqExtrCOO" ).EndTimer();
            }

            // Проверим соответствие решений всех методов!
            std::vector< std::pair<int, int> > solutionJVCdense;
            std::vector< std::pair<int, int> > solutionJVCsparse;
            std::vector< std::pair<int, int> > solutionMack;
            std::vector< std::pair<int, int> > solutionHungarian;

            for( int i = 0; i < n; i++ ) {
                if( actualJVCdense2N(i) < n ) {
                    solutionJVCdense.push_back( std::make_pair( i, actualJVCdense2N(i) ) );
                }
                if( actualJVCsparse2N(i) < n ) {
                    solutionJVCsparse.push_back( std::make_pair( i, actualJVCsparse2N(i) ) );
                }
                if( n <= boundN ) {
                    if( actualMack2N(i) < n ) {
                        solutionMack.push_back( std::make_pair( i, actualMack2N(i) ) );
                    }

                    if( actualHungarian2N(i) < n ) {
                        solutionHungarian.push_back( std::make_pair( i, actualHungarian2N(i) ) );
                    }
                }
            }

            if( ( solutionJVCdense.size() != solutionJVCsparse.size() ) ||
                ( ( n <= boundN ) && ( solutionJVCdense.size() != solutionMack.size() ) ) ||
                ( ( n <= boundN ) && ( solutionJVCdense.size() != solutionHungarian.size() ) ) )
            {
                counter_assign++;
            } else {
                int size = solutionJVCdense.size();
                for( int i = 0; i < size; i++ ) {
                    if( ( solutionJVCdense[i].first != solutionJVCsparse[i].first ) ||
                        ( solutionJVCdense[i].second != solutionJVCsparse[i].second ) ||
                        ( ( n <= boundN ) && ( ( solutionJVCdense[i].first != solutionMack[i].first ) ||
                        ( solutionJVCdense[i].second != solutionMack[i].second ) ) ) ||
                        ( ( n <= boundN ) && ( ( solutionJVCdense[i].first != solutionHungarian[i].first ) ||
                        ( solutionJVCdense[i].second != solutionHungarian[i].second ) ) ) )
                    {
                        counter_assign++;
                        break;
                    }
                }
            }

            // Оценим LAPCOST - суммарную стоимость назначения

            double totalcostJVCdense = 0.0;
            for( auto &elem : solutionJVCdense ) {
                totalcostJVCdense += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostJVCsparse = 0.0;
            for( auto &elem : solutionJVCsparse ) {
                totalcostJVCsparse += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostHungarian = 0.0;
            for( auto &elem : solutionHungarian ) {
                totalcostHungarian += mat_JVCdense( elem.first, elem.second );
            }
            double totalcostMack = 0.0;
            for( auto &elem : solutionMack ) {
                totalcostMack+= mat_JVCdense( elem.first, elem.second );
            }

            bool eq_totalcostJVCdense_totalcostJVCsparse = std::abs( totalcostJVCdense - totalcostJVCsparse ) < 1e-5;
            bool eq_totalcostJVCdense_totalcostHungarian = false;
            bool eq_totalcostJVCdense_totalcostMack = false;
            if( n <= boundN ) {
                eq_totalcostJVCdense_totalcostHungarian = std::abs( totalcostJVCdense - totalcostHungarian ) < 1e-5;
                eq_totalcostJVCdense_totalcostMack = std::abs( totalcostJVCdense - totalcostMack ) < 1e-5;
            }

            if( !eq_totalcostJVCdense_totalcostJVCsparse ||
                ( ( n <= boundN ) && ( !eq_totalcostJVCdense_totalcostMack ) ) ||
                ( ( n <= boundN ) && ( !eq_totalcostJVCdense_totalcostHungarian ) ) )
            {
                counter_lapcost++;
            }
        }
        if( print ) {
            for( auto &t : timer ) {
                std::cout << "timer " + t.first + " TimePerOp() = " << t.second.TimePerOp() << std::endl;
            }
        }

        for( auto &t : timer ) {
            if( ( ( n > boundN ) && ( t.first == "Hungarian" ) ) ||
                ( ( n > boundN ) && ( t.first == "SeqExtr" ) ) ||
                ( ( n > boundN ) && ( t.first == "SeqExtrCOO" ) ) ||
                ( ( n > boundN ) && ( t.first == "Mack" ) ) )
            {
                continue;
            }
            timeOfMethod.at( t.first ).push_back( timer.at( t.first ).TimePerOp() * 1.0e3 ); // мс
        }
        std::cout << "counter_lapcost=" << counter_lapcost << std::endl;
        std::cout << "counter_assign=" << counter_assign << std::endl;
        counter_lapcost_sum += counter_lapcost;
        counter_assign_sum += counter_assign;
    } // for n
    std::cout << "counter_lapcost_sum=" << counter_lapcost_sum << std::endl;
    std::cout << "counter_assign_sum=" << counter_assign_sum << std::endl;
#ifdef MATPLOTLIB
    if( print ) {
        std::cout << "plotting..." << std::endl;
    }
    for( auto &t : timeOfMethod ) {
        plt::plot(
            dimensionDouble.at( t.first ),
            ( t.second ), // Y
            estimated_keywords.at( t.first )
            );
    }
    plt::grid( true );
    plt::xlim( dimensionXlim.front(), dimensionXlim.back() );
    plt::legend();
    plt::save( "time2.png", dpi );
    if( show ) {
        plt::show();
    }
    plt::close();
#endif
#ifdef PRINTTOTXT
    // Оптимизировано для вывода в таблицу для документации
    std::ofstream os3;
    os3.open( "time_table.txt", std::ofstream::out );
    os3 << "| N ";
    for( int i = 0; i < dimensionXlim.size(); i++ ) {
        os3 << "| " << dimensionXlim[i] << " ";
    }
    os3 << "|" << std::endl;

    for( int i = 0; i <= dimensionXlim.size(); i++ ) {
        os3 << "| :-: ";
    }
    os3 << "|" << std::endl;

    std::vector<std::string> methods3 = { "Hungarian", "Mack", "SeqExtr", "SeqExtrCOO", "JVCdense", "JVCsparse" };

    for( auto &m : methods3 ) {
        os3 << "| " << m << " ";
        for( int i = 0; i < dimensionXlim.size(); i++ ) {
            for( auto &t : timeOfMethod ) {
                if( t.first == m ) {
                    if( i < ( ( t.second ).size() ) ) {
                        os3 << std::fixed << std::setprecision( 3 ) << "| " <<  ( ( t.second )[i] ) << " ";
                    } else {
                        os3 << "| - ";
                    }
                }
            }
        }
        os3 << "|" << std::endl;
    }
    os3.close();

    // Оптимизировано для вывода в таблицу для документации
    std::ofstream os4;
    os4.open( "time_table_relative.txt", std::ofstream::out );
    os4 << "| N ";
    for( int i = 0; i < dimensionXlim.size(); i++ ) {
        os4 << "| " << dimensionXlim[i] << " ";
    }
    os4 << "|" << std::endl;

    for( int i = 0; i <= dimensionXlim.size(); i++ ) {
        os4 << "| :-: ";
    }
    os4 << "|" << std::endl;

    std::vector<std::string> methods4 = { "Hungarian", "Mack", "SeqExtr", "SeqExtrCOO", "JVCdense" };

    for( auto &m : methods4 ) {
        os4 << "| " << m << " ";
        for( int i = 0; i < dimensionXlim.size(); i++ ) {
            for( auto &t : timeOfMethod ) {
                if( t.first == m ) {
                    if( i < ( ( t.second ).size() ) ) {
                        os4 << std::fixed << std::setprecision( 3 ) << "| " <<  ( ( t.second )[i] ) / ( timeOfMethod.at( "JVCsparse" ))[i] << " ";
                    } else {
                        os4 << "| - ";
                    }
                }
            }
        }
        os4 << "|" << std::endl;
    }
    os4.close();
#endif
}

BOOST_AUTO_TEST_CASE( test_spec )
{
    SPML::LAP::TSearchParam sp = SPML::LAP::TSearchParam::SP_Max;
    double resolution = 1.0e-6;

    double infValue = 1.0e7;
    double bigValue = 1.0e-6;
    double halfBigValue = 1.0e-5;

    int k = 4;//5;//
    int l = 5;
    arma::mat mat_dense = arma::mat( k, l, arma::fill::ones );
    mat_dense *= -bigValue;
    mat_dense(0,0) = 15;
    mat_dense(1,1) = 63;
    mat_dense(1,2) = 0.1;
    mat_dense(2,2) = 93;
    mat_dense(2,4) = 42;
    mat_dense(3,0) = 59;
    mat_dense(3,1) = 84;
    mat_dense(3,3) = 0.1;
//    mat_dense(4,3) = 74;

    mat_dense.print( "mat_dense" );

    arma::mat mat_dense_kl = arma::mat( ( k + l ), ( k + l ), arma::fill::ones );
    mat_dense_kl *= halfBigValue; //-bigValue;//
//    mat_dense_kl.print( "mat_dense_kl" );

    mat_dense_kl.submat( 0, 0, ( k - 1 ), ( l - 1 ) ) = mat_dense;
    mat_dense_kl.print( "mat_dense_kl" );

    arma::ivec actualJVCdense = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostJVCdense = 0.0;
    SPML::LAP::JVCdense( mat_dense_kl, ( k + l ), sp, infValue, resolution, actualJVCdense, lapcostJVCdense );

    arma::ivec actualMackdense = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostMackdense = 0.0;
    SPML::LAP::Mack( mat_dense_kl, ( k + l ), sp, infValue, resolution, actualMackdense, lapcostMackdense );

    arma::ivec actualHungdense = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostHungdense = 0.0;
    SPML::LAP::Hungarian( mat_dense_kl, ( k + l ), sp, infValue, resolution, actualHungdense, lapcostHungdense );

    arma::ivec actualSeqExtr = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostSeqExtr = 0.0;
    SPML::LAP::SequentalExtremum( mat_dense_kl, sp, infValue, resolution, actualSeqExtr, lapcostSeqExtr );

    arma::mat mat_dense_sp = arma::mat( k, l, arma::fill::zeros );
    for( int i = 0; i < k; i++ ) {
        for( int j = 0; j < l; j++ ) {
            if( mat_dense(i,j) > 0.0 ) {
                mat_dense_sp(i,j) = mat_dense(i,j);
            }
        }
    }
    mat_dense_sp.print( "mat_dense_sp" );

    arma::mat mat_diag_big = arma::mat( k, k, arma::fill::eye );
    mat_diag_big *= halfBigValue; // -bigValue;//
    arma::mat mat_for_sparse = arma::mat( k, ( l + k ), arma::fill::zeros );
    mat_for_sparse.submat( 0, 0, ( k - 1 ), ( l - 1 ) ) = mat_dense_sp;
    mat_for_sparse.submat( 0, l, ( k - 1 ), ( k + l - 1 ) ) = mat_diag_big;

    mat_for_sparse.print( "mat_for_sparse" );

    SPML::Sparse::CMatrixCSR mat_csr;
    SPML::Sparse::MatrixDenseToCSR( mat_for_sparse, mat_csr );

    arma::ivec actualJVCsparse = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostJVCsparse = 0.0;
    int resSparse = SPML::LAP::JVCsparse( mat_csr.csr_val, mat_csr.csr_kk, mat_csr.csr_first,
        sp, infValue, resolution, actualJVCsparse, lapcostJVCsparse );
    if( resSparse == 1 ) {
        assert( false );
    }

    arma::ivec actualJVCsparse2 = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostJVCsparse2 = 0.0;
    int resSparse2 = SPML::LAP::JVCsparse( mat_csr, sp, infValue, resolution, actualJVCsparse2, lapcostJVCsparse2 );
    if( resSparse2 == 1 ) {
        assert( false );
    }
    SPML::Sparse::CMatrixCOO mat_coo;
    SPML::Sparse::MatrixDenseToCOO( mat_for_sparse, mat_coo );

    arma::ivec actualSeqExtrCOO = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostSeqExtrCOO = 0.0;
    SPML::LAP::SequentalExtremum( mat_coo, sp, infValue, resolution, actualSeqExtrCOO, lapcostSeqExtrCOO );

    // Проверим соответствие решений всех методов!
    std::vector< std::pair<int, int> > solutionJVCdense;
    std::vector< std::pair<int, int> > solutionJVCsparse;
    std::vector< std::pair<int, int> > solutionJVCsparse2;
    std::vector< std::pair<int, int> > solutionMack;
    std::vector< std::pair<int, int> > solutionHung;
    std::vector< std::pair<int, int> > solutionSeqExtr;
    std::vector< std::pair<int, int> > solutionSeqExtrCOO;

    for( int i = 0; i < k; i++ ) {
        if( actualJVCdense(i) < l ) {
            solutionJVCdense.push_back( std::make_pair( i, actualJVCdense(i) ) );
        }
        if( actualJVCsparse(i) < l ) {
            solutionJVCsparse.push_back( std::make_pair( i, actualJVCsparse(i) ) );
        }
        if( actualJVCsparse2(i) < l ) {
            solutionJVCsparse2.push_back( std::make_pair( i, actualJVCsparse2(i) ) );
        }
        if( actualMackdense(i) < l ) {
            solutionMack.push_back( std::make_pair( i, actualMackdense(i) ) );
        }
        if( actualHungdense(i) < l ) {
            solutionHung.push_back( std::make_pair( i, actualHungdense(i) ) );
        }
        if( actualSeqExtr(i) < l ) {
            solutionSeqExtr.push_back( std::make_pair( i, actualSeqExtr(i) ) );
        }
        if( actualSeqExtrCOO(i) < l ) {
            solutionSeqExtrCOO.push_back( std::make_pair( i, actualSeqExtrCOO(i) ) );
        }
    }    

    BOOST_REQUIRE_EQUAL( solutionJVCdense.size(), solutionJVCsparse.size() );
    BOOST_REQUIRE_EQUAL( solutionJVCdense.size(), solutionJVCsparse2.size() );
    BOOST_REQUIRE_EQUAL( solutionJVCdense.size(), solutionMack.size() );
    BOOST_REQUIRE_EQUAL( solutionJVCdense.size(), solutionHung.size() );

    for( unsigned i = 0; i < solutionJVCdense.size(); i++ ) {
        BOOST_CHECK_EQUAL( solutionJVCdense[i].first, solutionJVCsparse[i].first );
        BOOST_CHECK_EQUAL( solutionJVCdense[i].second, solutionJVCsparse[i].second );

        BOOST_CHECK_EQUAL( solutionJVCdense[i].first, solutionJVCsparse2[i].first );
        BOOST_CHECK_EQUAL( solutionJVCdense[i].second, solutionJVCsparse2[i].second );

        BOOST_CHECK_EQUAL( solutionJVCdense[i].first, solutionMack[i].first );
        BOOST_CHECK_EQUAL( solutionJVCdense[i].second, solutionMack[i].second );

        BOOST_CHECK_EQUAL( solutionJVCdense[i].first, solutionHung[i].first );
        BOOST_CHECK_EQUAL( solutionJVCdense[i].second, solutionHung[i].second );
    }
}

BOOST_AUTO_TEST_CASE( test_spec2 )
{
    SPML::LAP::TSearchParam sp = SPML::LAP::TSearchParam::SP_Max;
    double resolution = 1.0e-7;

    double infValue = 1.0e7;//1.0e7;
    double bigValue = -1.0e6;
    double halfBigValue = -1.0e5;

//    double bigValue = -1.0e-2;
//    double halfBigValue = -1.0e-6;

//    double bigValue = 1.0e-6;
//    double halfBigValue = 1.0e-5;

    int k = 3;//5;//
    int l = 3;
    arma::mat mat_dense = arma::mat( k, l, arma::fill::ones );
    mat_dense *= bigValue;
    mat_dense(0,0) = 0.1;
    mat_dense(1,0) = 0.2;
    mat_dense(1,1) = 0.3;
    mat_dense(1,2) = 0.4;
    mat_dense(2,0) = 0.5;

//    arma::ivec actualJVCdense0 = arma::ivec( ( k + l ), arma::fill::zeros );
//    double lapcostJVCdense0 = 0.0;
//    SPML::LAP::JVCdense( mat_dense, 3, sp, infValue, resolution, actualJVCdense0, lapcostJVCdense0 );

//    mat_dense(0,0) = 15;
//    mat_dense(1,1) = 63;
//    mat_dense(1,2) = 0.1;
//    mat_dense(2,2) = 93;
//    mat_dense(2,4) = 42;
//    mat_dense(3,0) = 59;
//    mat_dense(3,1) = 84;
//    mat_dense(3,3) = 0.1;
////    mat_dense(4,3) = 74;

    mat_dense.print( "mat_dense" );

    arma::mat mat_dense_kl = arma::mat( ( k + l ), ( k + l ), arma::fill::ones );
    mat_dense_kl *= bigValue; //-bigValue;//

    arma::mat mat_dense_kk = arma::mat( k, k, arma::fill::ones );
    mat_dense_kk *= bigValue;
    arma::vec mat_dense_kk_diag = arma::vec( k, arma::fill::ones );
    mat_dense_kk_diag *= halfBigValue;//bigValue;//-
    mat_dense_kk.diag() = mat_dense_kk_diag;

    arma::mat mat_dense_ll = arma::mat( l, l, arma::fill::ones );
    mat_dense_ll *= bigValue;
//    arma::vec mat_dense_ll_diag = arma::vec( l, arma::fill::ones );
//    mat_dense_ll_diag *= halfBigValue;//bigValue;//-
//    mat_dense_ll.diag() = mat_dense_ll_diag;

    // Заполняем плотную матрицу:
    mat_dense_kl.submat( 0, 0, ( k - 1 ), ( l - 1 ) ) = mat_dense;
    mat_dense_kl.submat( k, 0, ( k + l - 1 ), ( l - 1 ) ) = mat_dense_ll;
    mat_dense_kl.submat( 0, l, ( k - 1 ), ( k + l - 1 ) ) = mat_dense_kk;
//    mat_dense_kl.submat( k, l, ( k + l - 1 ), ( k + l - 1 ) ) = mat_dense_kk;
    mat_dense_kl.print( "mat_dense_kl" );

    arma::ivec actualJVCdense = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostJVCdense = 0.0;
    SPML::LAP::JVCdense( mat_dense_kl, ( k + l ), sp, infValue, resolution, actualJVCdense, lapcostJVCdense );

    arma::ivec actualMackdense = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostMackdense = 0.0;
    SPML::LAP::Mack( mat_dense_kl, ( k + l ), sp, infValue, resolution, actualMackdense, lapcostMackdense );

    arma::ivec actualHungdense = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostHungdense = 0.0;
    SPML::LAP::Hungarian( mat_dense_kl, ( k + l ), sp, infValue, resolution, actualHungdense, lapcostHungdense );

    arma::ivec actualSeqExtr = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostSeqExtr = 0.0;
    SPML::LAP::SequentalExtremum( mat_dense_kl, sp, infValue, resolution, actualSeqExtr, lapcostSeqExtr );

    arma::mat mat_dense_sp = arma::mat( k, l, arma::fill::zeros );
    for( int i = 0; i < k; i++ ) {
        for( int j = 0; j < l; j++ ) {
            if( mat_dense(i,j) > 0.0 ) {
                mat_dense_sp(i,j) = mat_dense(i,j);
            }
        }
    }
    mat_dense_sp.print( "mat_dense_sp" );

    arma::mat mat_diag_big = arma::mat( k, k, arma::fill::eye );
    mat_diag_big *= halfBigValue; // -bigValue;//
    arma::mat mat_for_sparse = arma::mat( k, ( l + k ), arma::fill::zeros );
    mat_for_sparse.submat( 0, 0, ( k - 1 ), ( l - 1 ) ) = mat_dense_sp;
    mat_for_sparse.submat( 0, l, ( k - 1 ), ( k + l - 1 ) ) = mat_diag_big;

    mat_for_sparse.print( "mat_for_sparse" );

    SPML::Sparse::CMatrixCSR mat_csr;
    SPML::Sparse::MatrixDenseToCSR( mat_for_sparse, mat_csr );

    arma::ivec actualJVCsparse = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostJVCsparse = 0.0;
    int resSparse = SPML::LAP::JVCsparse( mat_csr.csr_val, mat_csr.csr_kk, mat_csr.csr_first,
        sp, infValue, resolution, actualJVCsparse, lapcostJVCsparse );
    if( resSparse == 1 ) {
        assert( false );
    }

    arma::ivec actualJVCsparse2 = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostJVCsparse2 = 0.0;
    int resSparse2 = SPML::LAP::JVCsparse( mat_csr, sp, infValue, resolution, actualJVCsparse2, lapcostJVCsparse2 );
    if( resSparse2 == 1 ) {
        assert( false );
    }
    SPML::Sparse::CMatrixCOO mat_coo;
    SPML::Sparse::MatrixDenseToCOO( mat_for_sparse, mat_coo );

    arma::ivec actualSeqExtrCOO = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostSeqExtrCOO = 0.0;
    SPML::LAP::SequentalExtremum( mat_coo, sp, infValue, resolution, actualSeqExtrCOO, lapcostSeqExtrCOO );

    // Проверим соответствие решений всех методов!
    std::vector< std::pair<int, int> > solutionJVCdense;
    std::vector< std::pair<int, int> > solutionJVCsparse;
    std::vector< std::pair<int, int> > solutionJVCsparse2;
    std::vector< std::pair<int, int> > solutionMack;
    std::vector< std::pair<int, int> > solutionHung;
    std::vector< std::pair<int, int> > solutionSeqExtr;
    std::vector< std::pair<int, int> > solutionSeqExtrCOO;

    for( int i = 0; i < k; i++ ) {
        if( actualJVCdense(i) < l ) {
            solutionJVCdense.push_back( std::make_pair( i, actualJVCdense(i) ) );
        }
        if( actualJVCsparse(i) < l ) {
            solutionJVCsparse.push_back( std::make_pair( i, actualJVCsparse(i) ) );
        }
        if( actualJVCsparse2(i) < l ) {
            solutionJVCsparse2.push_back( std::make_pair( i, actualJVCsparse2(i) ) );
        }
        if( actualMackdense(i) < l ) {
            solutionMack.push_back( std::make_pair( i, actualMackdense(i) ) );
        }
        if( actualHungdense(i) < l ) {
            solutionHung.push_back( std::make_pair( i, actualHungdense(i) ) );
        }
        if( actualSeqExtr(i) < l ) {
            solutionSeqExtr.push_back( std::make_pair( i, actualSeqExtr(i) ) );
        }
        if( actualSeqExtrCOO(i) < l ) {
            solutionSeqExtrCOO.push_back( std::make_pair( i, actualSeqExtrCOO(i) ) );
        }
    }

    BOOST_REQUIRE_EQUAL( solutionJVCdense.size(), solutionJVCsparse.size() );
    BOOST_REQUIRE_EQUAL( solutionJVCdense.size(), solutionJVCsparse2.size() );
    BOOST_REQUIRE_EQUAL( solutionJVCdense.size(), solutionMack.size() );
    BOOST_REQUIRE_EQUAL( solutionJVCdense.size(), solutionHung.size() );

    for( unsigned i = 0; i < solutionJVCdense.size(); i++ ) {
        BOOST_CHECK_EQUAL( solutionJVCdense[i].first, solutionJVCsparse[i].first );
        BOOST_CHECK_EQUAL( solutionJVCdense[i].second, solutionJVCsparse[i].second );

        BOOST_CHECK_EQUAL( solutionJVCdense[i].first, solutionJVCsparse2[i].first );
        BOOST_CHECK_EQUAL( solutionJVCdense[i].second, solutionJVCsparse2[i].second );

        BOOST_CHECK_EQUAL( solutionJVCdense[i].first, solutionMack[i].first );
        BOOST_CHECK_EQUAL( solutionJVCdense[i].second, solutionMack[i].second );

        BOOST_CHECK_EQUAL( solutionJVCdense[i].first, solutionHung[i].first );
        BOOST_CHECK_EQUAL( solutionJVCdense[i].second, solutionHung[i].second );
    }
}
