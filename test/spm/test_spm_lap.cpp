//----------------------------------------------------------------------------------------------------------------------
///
/// \file       test_spm_lap.cpp
/// \brief      Тестирование задач о назначениях
///

#define BOOST_TEST_MODULE test_spm_lap

#include <boost/test/included/unit_test.hpp>
#include <vector>
#include <set>
#include <map>
#include <armadillo>
#include <random>
#include <numeric>
#include <fstream>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <thread>
#include <chrono>

#include "spm/timing/timing.hpp"
#include "spm/lap/lap.hpp"
#include "spm/lap/sparse.hpp"
#include "spm/lap/murty.hpp"

#ifdef MATPLOTLIB
    #include <matplotlibcpp.h>
#endif

bool show = false;//true;//

//----------------------------------------------------------------------------------------------------------------------
// Тестовые задачи:
//----------------------------------------------------------------------------------------------------------------------
// 0
arma::mat mat_0_dense = {
   { 0.6, 0.3, 0.0, 0.1, 0.0, 0.0 },
   { 0.0, 0.2, 0.7, 0.0, 0.1, 0.0 },
   { 0.3, 0.5, 0.1, 0.0, 0.0, 0.1 }
};

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
// 4
arma::mat mat_4_dense_for_max = {
    { 93.0, -1e6, 91.0, -1e6, -1e6, -1e5, -1e6, -1e6 },
    { -1e6, 93.0, 90.0, -1e6, 98.0, -1e6, -1e5, -1e6 },
    { -1e6, -1e6, 91.0, 90.0, -1e6, -1e6, -1e6, -1e5 },
    { -1e6, -1e6, -1e6, -1e6, -1e6, -1e6, -1e6, -1e6 },
    { -1e6, -1e6, -1e6, -1e6, -1e6, -1e6, -1e6, -1e6 },
    { -1e6, -1e6, -1e6, -1e6, -1e6, -1e6, -1e6, -1e6 },
    { -1e6, -1e6, -1e6, -1e6, -1e6, -1e6, -1e6, -1e6 },
    { -1e6, -1e6, -1e6, -1e6, -1e6, -1e6, -1e6, -1e6 }
};
arma::mat mat_4_dense_for_min = {
    { 93.0, 1e6, 91.0,  1e6,  1e6, 1e5, 1e6, 1e6 },
    { 1e6, 93.0, 90.0,  1e6, 98.0, 1e6, 1e5, 1e6 },
    { 1e6,  1e6, 91.0, 90.0,  1e6, 1e6, 1e6, 1e5 },
    { 1e6,  1e6,  1e6,  1e6,  1e6, 1e6, 1e6, 1e6 },
    { 1e6,  1e6,  1e6,  1e6,  1e6, 1e6, 1e6, 1e6 },
    { 1e6,  1e6,  1e6,  1e6,  1e6, 1e6, 1e6, 1e6 },
    { 1e6,  1e6,  1e6,  1e6,  1e6, 1e6, 1e6, 1e6 },
    { 1e6,  1e6,  1e6,  1e6,  1e6, 1e6, 1e6, 1e6 }
};
arma::ivec expected_4_max = { 0, 4, 2 };
arma::ivec expected_4_min = { 0, 2, 3 };
//----------------------------------------------------------------------------------------------------------------------
// 5 - test matrix from murty1968.pdf
arma::mat mat_5_dense = {
    { 7,  51, 52, 87, 38, 60, 74, 66, 0,  20 },
    { 50, 12, 0,  64, 8,  53, 0,  46, 76, 42 },
    { 27, 77, 0,  18, 22, 48, 44, 13, 0,  57 },
    { 62, 0,  3,  8,  5,  6,  14, 0,  26, 39 },
    { 0,  97, 0,  5,  13,  0, 41, 31, 62, 48 },
    { 79, 68, 0,  0,  15, 12, 17, 47, 35, 43 },
    { 76, 99, 48, 27, 34, 0,  0,  0,  28, 0 },
    { 0,  20, 9,  27, 46, 15, 84, 19, 3,  24 },
    { 56, 10, 45, 39, 0,  93, 67, 79, 19, 38 },
    { 27, 0,  39, 53, 46, 24, 69, 46, 23, 1 }
};
arma::ivec expected_5_min_1_best = { 8, 6, 2, 7, 5, 3, 9, 0, 4, 1 };
arma::ivec expected_5_min_2_best = { 8, 6, 2, 1, 5, 3, 7, 0, 4, 9 };

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_5 )

BOOST_AUTO_TEST_CASE( test_mat_5_murty_min )
{
    SPM::Sparse::CMatrixCSR csr;
    double empty = 1.0e12;
    SPM::Sparse::MatrixDenseToCSR( mat_5_dense, csr, empty );

    arma::ivec rowsol = arma::ivec( csr.n_cols(), arma::fill::zeros );
    double resolution = 1e-7;
    const auto sp = SPM::LAP::TSearchParam::SP_Min;
    double infValue = 1e7;

    SPM::LAP::Murty murty( csr, sp, infValue, resolution );
    SPM::LAP::MurtySolution out;

    std::vector<double> time;

    for( int i = 0; i < 2; i++ ) {
        SPM::Timing::CTimeKeeper tk;
        tk.StartTimer();
        bool ok = murty.findNext( out );
        tk.EndTimer();
        if( ok ) {
            double eps = 1e-7;
            if( i == 0 ) {
                BOOST_CHECK_EQUAL( arma::approx_equal( out.x, expected_5_min_1_best, "absdiff", eps ), true );
            } else if( i == 1 ) {
                BOOST_CHECK_EQUAL( arma::approx_equal( out.x, expected_5_min_2_best, "absdiff", eps ), true );
            }
            time.push_back( tk.TimeSumm() );
            std::cout << "i = " << i << "; cost = " << out.cost << "; sol = " << out.x.t();
        } else {
            std::cout << "break!" << std::endl;
            break;
        }
    }
    for( const auto & t : time ) {
        std::cout << "time [mcs] = " << ( t * 1000000 ) << std::endl;
    }
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_0 )

BOOST_AUTO_TEST_CASE( test_mat_0_murty_max )
{
    SPM::Sparse::CMatrixCSR csr;
    SPM::Sparse::MatrixDenseToCSR( mat_0_dense, csr );

    arma::ivec rowsol = arma::ivec( csr.n_cols(), arma::fill::zeros );
    double resolution = 1e-7;
    const auto sp = SPM::LAP::TSearchParam::SP_Max;
    double infValue = 1e7;

    SPM::LAP::Murty murty( csr, sp, infValue, resolution );
    SPM::LAP::MurtySolution out;

    std::vector<double> time;

    for( int i = 0; i < 100; i++ ) {
        SPM::Timing::CTimeKeeper tk;
        tk.StartTimer();
        bool ok = murty.findNext( out );
        tk.EndTimer();
        if( ok ) {
            time.push_back( tk.TimeSumm() );
            std::cout << "i = " << i << "; cost = " << out.cost << "; sol = " << out.x.t();
        } else {
            std::cout << "break!" << std::endl;
            break;
        }
    }
    for( const auto & t : time ) {
        std::cout << "time [mcs] = " << ( t * 1000000 ) << std::endl;
    }
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_1_min )

BOOST_AUTO_TEST_CASE( test_mat_1_seqextr_min )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;//mat_1_dense.max();
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::SequentalExtremum( mat_1_dense, SPM::LAP::TSearchParam::SP_Min, infValue, resolution, actual, lapcost );
    double eps = 1e-7;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_subextr_min, "absdiff", eps ), true );

    arma::ivec actualCOO = arma::ivec( size, arma::fill::zeros );
    double lapcostCOO;
    SPM::Sparse::CMatrixCOO coo;
    SPM::Sparse::MatrixDenseToCOO( mat_1_dense, coo );
    SPM::LAP::SequentalExtremum( coo, SPM::LAP::TSearchParam::SP_Min, infValue, resolution, actualCOO, lapcostCOO );
    BOOST_CHECK_EQUAL( arma::approx_equal( actualCOO, expected_1_subextr_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_1_jvc_dense_min )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::JVCdense( mat_1_dense, size, SPM::LAP::TSearchParam::SP_Min, infValue, resolution, actual, lapcost );
    double eps = 1e-7;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_1_mack_min )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::Mack( mat_1_dense, size, SPM::LAP::TSearchParam::SP_Min, infValue, resolution, actual, lapcost );
    double eps = 1e-7;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_1_hungarian_min )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::Hungarian( mat_1_dense, size, SPM::LAP::TSearchParam::SP_Min, infValue, resolution, actual, lapcost );
    double eps = 1e-7;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_1_max )

BOOST_AUTO_TEST_CASE( test_mat_1_seqextr_max )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;//mat_1_dense.max();
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::SequentalExtremum( mat_1_dense, SPM::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );
    double eps = 1e-7;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_subextr_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_1_jvc_dense_max )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::JVCdense( mat_1_dense, size, SPM::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );
    double eps = 1e-7;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_1_mack_max )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::Mack( mat_1_dense, size, SPM::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );
    double eps = 1e-7;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_1_hungarian_max )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::Hungarian( mat_1_dense, size, SPM::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );
    double eps = 1e-7;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_2_min )

BOOST_AUTO_TEST_CASE( test_mat_2_seqextr_min )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;//mat_1_dense.max();
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::SequentalExtremum( mat_2_dense, SPM::LAP::TSearchParam::SP_Min, infValue, resolution, actual, lapcost );
    double eps = 1e-7;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_subextr_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_2_jvc_dense_min )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::JVCdense( mat_2_dense, size, SPM::LAP::TSearchParam::SP_Min, infValue, resolution, actual, lapcost );
    double eps = 1e-7;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_2_mack_min )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::Mack( mat_2_dense, size, SPM::LAP::TSearchParam::SP_Min, infValue, resolution, actual, lapcost );
    double eps = 1e-7;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_2_hungarian_min )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::Hungarian( mat_2_dense, size, SPM::LAP::TSearchParam::SP_Min, infValue, resolution, actual, lapcost );
    double eps = 1e-7;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_2_max )

BOOST_AUTO_TEST_CASE( test_mat_2_seqextr_max )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;//mat_1_dense.max();
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::SequentalExtremum( mat_2_dense, SPM::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );
    double eps = 1e-7;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_subextr_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_2_jvc_dense_max )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::JVCdense( mat_2_dense, size, SPM::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );
    double eps = 1e-7;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_2_mack_max )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::Mack( mat_2_dense, size, SPM::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );
    double eps = 1e-7;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_2_hungarian_max )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;//mat_1_dense.max();
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::Hungarian( mat_2_dense, size, SPM::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );
    double eps = 1e-7;
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
    SPM::Sparse::MatrixDenseToCSR( mat_3_dense, csr_val, csr_kk, csr_first );

    unsigned size = mat_3_dense.n_cols;
    arma::ivec actual1 = arma::ivec( size, arma::fill::zeros );
    arma::ivec actual2 = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;
    double resolution = 1e-7;
    double lapcost1, lapcost2;
    SPM::LAP::JVCdense( mat_3_dense, size, SPM::LAP::TSearchParam::SP_Max, infValue, resolution, actual1, lapcost1 );
    SPM::LAP::JVCsparse( csr_val, csr_kk, csr_first, SPM::LAP::TSearchParam::SP_Max, infValue, resolution, actual2, lapcost2 );

    double eps = 1e-7;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual1, expected_3_max, "absdiff", eps ), true );
    BOOST_CHECK_EQUAL( arma::approx_equal( actual2, expected_3_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_4_max )

BOOST_AUTO_TEST_CASE( test_mat_4_hungarian_max )
{
    unsigned size = mat_4_dense_for_max.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::Hungarian( mat_4_dense_for_max, size, SPM::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );

    double eps = 1e-7;
    actual.resize( 3 );
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_4_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_4_mack_max )
{
    unsigned size = mat_4_dense_for_max.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::Mack( mat_4_dense_for_max, size, SPM::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );

    double eps = 1e-7;
    actual.resize( 3 );
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_4_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_4_jvc_dense_max )
{
    unsigned size = mat_4_dense_for_max.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::JVCdense( mat_4_dense_for_max, size, SPM::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );

    double eps = 1e-7;
    actual.resize( 3 );
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_4_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_4_jvc_sparse_max )
{
    SPM::Sparse::CMatrixCOO coo;
    for( int i = 0; i < mat_4_dense_for_max.n_rows; i++ ) {
        for( int j = 0; j < mat_4_dense_for_max.n_cols; j++ ) {
            if( mat_4_dense_for_max( i, j ) > -1e6 ) {
                coo.coo_val.push_back( mat_4_dense_for_max( i, j ) );
                coo.coo_row.push_back( i );
                coo.coo_col.push_back( j );
            }
        }
    }
    SPM::Sparse::CMatrixCSR csr;
    SPM::Sparse::MatrixCOOtoCSR( coo, csr );

    unsigned size = mat_4_dense_for_max.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::JVCsparse( csr, SPM::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );

    double eps = 1e-7;
    actual.resize( 3 );
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_4_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_4_min )

BOOST_AUTO_TEST_CASE( test_mat_4_hungarian_min )
{
    unsigned size = mat_4_dense_for_min.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::Hungarian( mat_4_dense_for_min, size, SPM::LAP::TSearchParam::SP_Min, infValue, resolution, actual, lapcost );

    double eps = 1e-7;
    actual.resize( 3 );
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_4_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_4_mack_min )
{
    unsigned size = mat_4_dense_for_min.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::Mack( mat_4_dense_for_min, size, SPM::LAP::TSearchParam::SP_Min, infValue, resolution, actual, lapcost );

    double eps = 1e-7;
    actual.resize( 3 );
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_4_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_4_jvc_dense_min )
{
    unsigned size = mat_4_dense_for_min.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::JVCdense( mat_4_dense_for_min, size, SPM::LAP::TSearchParam::SP_Min, infValue, resolution, actual, lapcost );

    double eps = 1e-7;
    actual.resize( 3 );
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_4_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_4_jvc_sparse_min )
{
    SPM::Sparse::CMatrixCOO coo;
    for( int i = 0; i < mat_4_dense_for_min.n_rows; i++ ) {
        for( int j = 0; j < mat_4_dense_for_min.n_cols; j++ ) {
            if( mat_4_dense_for_min( i, j ) < 1e6 ) {
                coo.coo_val.push_back( mat_4_dense_for_min( i, j ) );
                coo.coo_row.push_back( i );
                coo.coo_col.push_back( j );
            }
        }
    }
    SPM::Sparse::CMatrixCSR csr;
    SPM::Sparse::MatrixCOOtoCSR( coo, csr );

    unsigned size = mat_4_dense_for_min.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double infValue = 1e7;
    double resolution = 1e-7;
    double lapcost;
    SPM::LAP::JVCsparse( csr, SPM::LAP::TSearchParam::SP_Min, infValue, resolution, actual, lapcost );

    double eps = 1e-7;
    actual.resize( 3 );
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_4_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

//BOOST_AUTO_TEST_SUITE( test_mat_5_max )

//BOOST_AUTO_TEST_CASE( test_mat_5_jvc_sparse_max )
//{
//    SPM::Sparse::CMatrixCOO coo;
//    for( int i = 0; i < mat_5_dense_for_max.n_rows; i++ ) {
//        for( int j = 0; j < mat_5_dense_for_max.n_cols; j++ ) {
//            if( mat_5_dense_for_max( i, j ) > -1e6 ) {
//                coo.coo_val.push_back( mat_5_dense_for_max( i, j ) );
//                coo.coo_row.push_back( i );
//                coo.coo_col.push_back( j );
//            }
//        }
//    }
//    SPM::Sparse::CMatrixCSR csr;
//    SPM::Sparse::MatrixCOOtoCSR( coo, csr );

//    unsigned size = mat_5_dense_for_max.n_cols;
//    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
//    double infValue = 1e7;
//    double resolution = 1e-7;
//    double lapcost;
//    SPM::LAP::JVCsparse( csr, SPM::LAP::TSearchParam::SP_Max, infValue, resolution, actual, lapcost );

//    double eps = 1e-7;
//    actual.resize( 3 );
//    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_4_max, "absdiff", eps ), true );
//}

//BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE( test_spec )
{
    SPM::LAP::TSearchParam sp = SPM::LAP::TSearchParam::SP_Max;
    double resolution = 1e-7;

    double infValue = 1e7;
    double bigValue = -1e6;
    double halfBigValue = -1e5;

    int k = 4;//5;//
    int l = 5;
    arma::mat mat_dense = arma::mat( k, l, arma::fill::ones );
    mat_dense *= bigValue;
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
    mat_dense_kl *= halfBigValue;

    mat_dense_kl.submat( 0, 0, ( k - 1 ), ( l - 1 ) ) = mat_dense;
    mat_dense_kl.print( "mat_dense_kl" );

    arma::ivec actualJVCdense = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostJVCdense = 0.0;
    SPM::LAP::JVCdense( mat_dense_kl, ( k + l ), sp, infValue, resolution, actualJVCdense, lapcostJVCdense );

    arma::ivec actualMackdense = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostMackdense = 0.0;
    SPM::LAP::Mack( mat_dense_kl, ( k + l ), sp, infValue, resolution, actualMackdense, lapcostMackdense );

    arma::ivec actualHungdense = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostHungdense = 0.0;
    SPM::LAP::Hungarian( mat_dense_kl, ( k + l ), sp, infValue, resolution, actualHungdense, lapcostHungdense );

    arma::ivec actualSeqExtr = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostSeqExtr = 0.0;
    SPM::LAP::SequentalExtremum( mat_dense_kl, sp, infValue, resolution, actualSeqExtr, lapcostSeqExtr );

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

    SPM::Sparse::CMatrixCSR mat_csr;
    SPM::Sparse::MatrixDenseToCSR( mat_for_sparse, mat_csr );

    arma::ivec actualJVCsparse = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostJVCsparse = 0.0;
    int resSparse = SPM::LAP::JVCsparse( mat_csr.csr_val, mat_csr.csr_kk, mat_csr.csr_first,
        sp, infValue, resolution, actualJVCsparse, lapcostJVCsparse );
    if( resSparse == 1 ) {
        assert( false );
    }

    arma::ivec actualJVCsparse2 = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostJVCsparse2 = 0.0;
    int resSparse2 = SPM::LAP::JVCsparse( mat_csr, sp, infValue, resolution, actualJVCsparse2, lapcostJVCsparse2 );
    if( resSparse2 == 1 ) {
        assert( false );
    }
    SPM::Sparse::CMatrixCOO mat_coo;
    SPM::Sparse::MatrixDenseToCOO( mat_for_sparse, mat_coo );

    arma::ivec actualSeqExtrCOO = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostSeqExtrCOO = 0.0;
    SPM::LAP::SequentalExtremum( mat_coo, sp, infValue, resolution, actualSeqExtrCOO, lapcostSeqExtrCOO );

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
    SPM::LAP::TSearchParam sp = SPM::LAP::TSearchParam::SP_Max;
    double resolution = 1e-7;

    double infValue = 1e7;
    double bigValue = -1e6;
    double halfBigValue = -1e5;

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
//    SPM::LAP::JVCdense( mat_dense, 3, sp, infValue, resolution, actualJVCdense0, lapcostJVCdense0 );

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
    SPM::LAP::JVCdense( mat_dense_kl, ( k + l ), sp, infValue, resolution, actualJVCdense, lapcostJVCdense );

    arma::ivec actualMackdense = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostMackdense = 0.0;
    SPM::LAP::Mack( mat_dense_kl, ( k + l ), sp, infValue, resolution, actualMackdense, lapcostMackdense );

    arma::ivec actualHungdense = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostHungdense = 0.0;
    SPM::LAP::Hungarian( mat_dense_kl, ( k + l ), sp, infValue, resolution, actualHungdense, lapcostHungdense );

    arma::ivec actualSeqExtr = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostSeqExtr = 0.0;
    SPM::LAP::SequentalExtremum( mat_dense_kl, sp, infValue, resolution, actualSeqExtr, lapcostSeqExtr );

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

    SPM::Sparse::CMatrixCSR mat_csr;
    SPM::Sparse::MatrixDenseToCSR( mat_for_sparse, mat_csr );

    arma::ivec actualJVCsparse = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostJVCsparse = 0.0;
    int resSparse = SPM::LAP::JVCsparse( mat_csr.csr_val, mat_csr.csr_kk, mat_csr.csr_first,
        sp, infValue, resolution, actualJVCsparse, lapcostJVCsparse );
    if( resSparse == 1 ) {
        assert( false );
    }

    arma::ivec actualJVCsparse2 = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostJVCsparse2 = 0.0;
    int resSparse2 = SPM::LAP::JVCsparse( mat_csr, sp, infValue, resolution, actualJVCsparse2, lapcostJVCsparse2 );
    if( resSparse2 == 1 ) {
        assert( false );
    }
    SPM::Sparse::CMatrixCOO mat_coo;
    SPM::Sparse::MatrixDenseToCOO( mat_for_sparse, mat_coo );

    arma::ivec actualSeqExtrCOO = arma::ivec( ( k + l ), arma::fill::zeros );
    double lapcostSeqExtrCOO = 0.0;
    SPM::LAP::SequentalExtremum( mat_coo, sp, infValue, resolution, actualSeqExtrCOO, lapcostSeqExtrCOO );

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

BOOST_AUTO_TEST_CASE( exact_methods_match_exhaustive_permutations )
{
    std::mt19937 gen( 114 );
    for( int trial = 0; trial < 30; ++trial ) {
        const int n = 2 + gen() % 4;
        arma::mat matrix( n, n );
        for( double &value : matrix ) {
            value = ( static_cast<int>( gen() % 41 ) - 20 ) * 0.25;
        }
        for( auto sp : { SPM::LAP::SP_Min, SPM::LAP::SP_Max } ) {
            std::vector<int> perm( n );
            std::iota( perm.begin(), perm.end(), 0 );
            double expected = ( sp == SPM::LAP::SP_Min ) ? 1e12 : -1e12;
            do {
                double value = 0;
                for( int i = 0; i < n; ++i ) {
                    value += matrix(i, perm[i]);
                }
                expected = ( sp == SPM::LAP::SP_Min ) ? std::min( expected, value ) : std::max( expected, value );
            } while( std::next_permutation( perm.begin(), perm.end() ) );
            for( int method = 0; method < 4; ++method ) {
                arma::ivec x( n );
                double actual = 0;
                if( method == 0 ) {
                    SPM::LAP::JVCdense( matrix, n, sp, 1e12, 1e-9, x, actual );
                } else if( method == 1 ) {
                    SPM::LAP::Mack( matrix, n, sp, 1e12, 1e-9, x, actual );
                } else if( method == 2 ) {
                    SPM::LAP::Hungarian( matrix, n, sp, 1e12, 1e-9, x, actual );
                } else {
                    SPM::Sparse::CMatrixCSR csr;
                    SPM::Sparse::MatrixDenseToCSR( matrix, csr, 1e30, 1e-12 );
                    BOOST_REQUIRE_EQUAL( SPM::LAP::JVCsparse( csr, sp, 1e12, 1e-9, x, actual ), 0 );
                }
                BOOST_CHECK_SMALL( actual - expected, 1e-7 );
                std::set<int> columns;
                double value = 0;
                for( int i = 0; i < n; ++i ) {
                    BOOST_REQUIRE_GE( x(i), 0 );
                    BOOST_REQUIRE_LT( x(i), n );
                    BOOST_CHECK( columns.insert( x(i) ).second );
                    value += matrix(i, x(i));
                }
                BOOST_CHECK_SMALL( actual - value, 1e-7 );
            }
        }
    }
}
