//----------------------------------------------------------------------------------------------------------------------
///
/// \file       test_lap.cpp
/// \brief      Тестирование задач о назначениях
/// \date       18.05.22 - создан
/// \author     Соболев А.А.
///

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_lap

#include <boost/test/unit_test.hpp>
#include <vector>
#include <map>
#include <matplotlibcpp.h>

#include <lap.h>
#include <timing.h>
#include <armadillo>


BOOST_AUTO_TEST_CASE( size5to50 )
{
    bool print = true; //false; //

    int cycle_count = 5000;

    namespace plt = matplotlibcpp;
    double width = 100;
    double height = 50;

    const double in2mm = 25.4;// mm (fixed)
//    const double pt2mm = 0.3528;// mm (fixed)
    const double dpi = 300;// dpi (variable)
    const double mm2px = dpi / in2mm;//
    size_t pixels_width = std::round( width * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
    plt::title( "Время решения задачи о назначениях" );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, [мс]" );

    std::vector<int> dimensionXlim = { 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 };
    std::vector<double> dimensionLong( dimensionXlim.begin(), dimensionXlim.end() );
    std::vector<double> dimensionShort( dimensionXlim.begin(), dimensionXlim.end() );
    int boundN = dimensionShort.back();

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "JVC", dimensionLong },
        { "Mack", dimensionLong },
        { "Hungarian", dimensionShort }
    };

    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "JVC", {} },
        { "Mack",{} },
        { "Hungarian",{} }
    };

    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
        { "JVC", { { "color", "red" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVC" } } },
        { "Mack", { { "color", "green" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "blue" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Hungarian" } } }
    };

    for( auto &n : dimensionXlim ) {

        arma::mat p_JV( n, n, arma::fill::randn );
        arma::mat p_Mack( n, n, arma::fill::randn );
        arma::mat p_Hungarian( n, n, arma::fill::randn );
        arma::ivec actualJV = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );

        std::map<std::string, SPML::Timing::CTimeKeeper> timer = {
            { "JVC", SPML::Timing::CTimeKeeper() },
            { "Mack", SPML::Timing::CTimeKeeper() },
            { "Hungarian", SPML::Timing::CTimeKeeper() }
        };

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }
            arma::arma_rng::set_seed( cycle );
            p_JV.randn();
            arma::arma_rng::set_seed( cycle );
            p_Mack.randn();
            arma::arma_rng::set_seed( cycle );
            p_Hungarian.randn();

            double maxcost = p_JV.max();
            double resolution = 1e-6;

            timer.at( "JVC" ).StartTimer();
            LAP::CAssignmentProblemSolver::JVC( p_JV, n, LAP::TSearchParam::Max, maxcost, resolution, actualJV );
            timer.at( "JVC" ).EndTimer();

            timer.at( "Mack" ).StartTimer();
            LAP::CAssignmentProblemSolver::Mack( p_Mack, n, LAP::TSearchParam::Max, actualMack );
            timer.at( "Mack" ).EndTimer();

            if( n <= boundN ) {
                timer.at( "Hungarian" ).StartTimer();
                LAP::CAssignmentProblemSolver::Hungarian( p_Hungarian, n, LAP::TSearchParam::Max, actualHungarian );
                timer.at( "Hungarian" ).EndTimer();
            }

            for( int i = 0; i < n; i++ ) {
                BOOST_CHECK_EQUAL( ( abs( actualJV(i) - actualMack(i) ) < 1e-5 ), true );
                if( n <= boundN ) {
                    BOOST_CHECK_EQUAL( ( abs( actualJV(i) - actualHungarian(i) ) < 1e-5 ), true );
                }
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
    plt::save( "time_2d_assignment_copmare_N_from_2_to_50.png", dpi );
    plt::show();
    plt::close();
    plt::clf();
    plt::cla();
}

BOOST_AUTO_TEST_CASE( size50to1000 )
{
    bool print = true; //false; //

    int cycle_count = 50;

    namespace plt = matplotlibcpp;
    double width = 100;
    double height = 50;

    const double in2mm = 25.4;// mm (fixed)
//    const double pt2mm = 0.3528;// mm (fixed)
    const double dpi = 300;// dpi (variable)
    const double mm2px = dpi / in2mm;//
    size_t pixels_width = std::round( width * mm2px);//
    size_t pixels_height = std::round( height * mm2px);//
    plt::figure_size( pixels_width, pixels_height );
    plt::title( "Время решения задачи о назначениях" );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, [мс]" );

    std::vector<int> dimensionXlim =     { 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000 };
    std::vector<double> dimensionLong =  { 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000 };
    std::vector<double> dimensionShort = { 50, 100, 150, 200, 250, 300, 350, 400, 450, 500 };

    int boundN = dimensionShort.back();

    std::map<std::string, std::vector<double>> dimensionDouble = {
        { "JVC", dimensionLong },
        { "Mack", dimensionLong },
        { "Hungarian", dimensionShort }
    };

    std::map<std::string, std::vector<double>> timeOfMethod = {
        { "JVC", {} },
        { "Mack",{} },
        { "Hungarian",{} }
    };

    std::map<std::string, std::map<std::string, std::string>> estimated_keywords = {
        { "JVC", { { "color", "red" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "JVC" } } },
        { "Mack", { { "color", "green" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Mack" } } },
        { "Hungarian", { { "color", "blue" }, {"marker", "o"}, { "linestyle", "-" }, { "linewidth", "1" }, { "label", "Hungarian" } } }
    };

    for( auto &n : dimensionXlim ) {

        arma::mat p_JV( n, n, arma::fill::randn );
        arma::mat p_Mack( n, n, arma::fill::randn );
        arma::mat p_Hungarian( n, n, arma::fill::randn );
        arma::ivec actualJV = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );

        std::map<std::string, SPML::Timing::CTimeKeeper> timer = {
            { "JVC", SPML::Timing::CTimeKeeper() },
            { "Mack", SPML::Timing::CTimeKeeper() },
            { "Hungarian", SPML::Timing::CTimeKeeper() }
        };

        for( int cycle = 0; cycle < cycle_count; cycle++ ) {
            if( print ) {
                std::cout << "cycle = " << ( cycle + 1 ) << "/" << cycle_count << " dim = " << n << std::endl;
            }
            arma::arma_rng::set_seed( cycle );
            p_JV.randn();
            arma::arma_rng::set_seed( cycle );
            p_Mack.randn();
            arma::arma_rng::set_seed( cycle );
            p_Hungarian.randn();

            double maxcost = p_JV.max();
            double resolution = 1e-6;

            timer.at( "JVC" ).StartTimer();
            LAP::CAssignmentProblemSolver::JVC( p_JV, n, LAP::TSearchParam::Max, maxcost, resolution, actualJV );
            timer.at( "JVC" ).EndTimer();

            timer.at( "Mack" ).StartTimer();
            LAP::CAssignmentProblemSolver::Mack( p_Mack, n, LAP::TSearchParam::Max, actualMack );
            timer.at( "Mack" ).EndTimer();

            if( n <= boundN ) {
                timer.at( "Hungarian" ).StartTimer();
                LAP::CAssignmentProblemSolver::Hungarian( p_Hungarian, n, LAP::TSearchParam::Max, actualHungarian );
                timer.at( "Hungarian" ).EndTimer();
            }

            for( int i = 0; i < n; i++ ) {
                BOOST_CHECK_EQUAL( ( abs( actualJV(i) - actualMack(i) ) < 1e-5 ), true );
                if( n <= boundN ) {
                    BOOST_CHECK_EQUAL( ( abs( actualJV(i) - actualHungarian(i) ) < 1e-5 ), true );
                }
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
    plt::save( "time_2d_assignment_copmare_N_from_5_to_1000.png", dpi );
    plt::show();
    plt::close();
    plt::clf();
    plt::cla();
//    Py_Finalize();
}

//----------------------------------------------------------------------------------------------------------------------
// Тестовые задачи:

arma::mat mat_1 = {
    { 7.0, 2.0, 1.0, 9.0, 4.0 },
    { 9.0, 6.0, 9.0, 6.0, 5.0 },
    { 3.0, 8.0, 3.0, 1.0, 8.0 },
    { 7.0, 9.0, 4.0, 2.0, 2.0 },
    { 8.0, 4.0, 7.0, 4.0, 8.0 }
};
arma::ivec expected_1_min = { 2, 4, 0, 3, 1 };
arma::ivec expected_1_max = { 3, 2, 4, 1, 0 };

arma::mat mat_2 = {
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

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_1_min )

BOOST_AUTO_TEST_CASE( test_mat_1_jvc_min )
{
    int size = mat_1.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double maxcost = mat_1.max();
    double resolution = 1e-6;
    LAP::CAssignmentProblemSolver::JVC( mat_1, size, LAP::TSearchParam::Min, maxcost, resolution, actual );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_1_mack_min )
{
    int size = mat_1.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    LAP::CAssignmentProblemSolver::Mack( mat_1, size, LAP::TSearchParam::Min, actual );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_1_hungarian_min )
{
    int size = mat_1.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    LAP::CAssignmentProblemSolver::Hungarian( mat_1, size, LAP::TSearchParam::Min, actual );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_1_max )

BOOST_AUTO_TEST_CASE( test_mat_1_jvc_max )
{
    int size = mat_1.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double maxcost = mat_1.max();
    double resolution = 1e-6;
    LAP::CAssignmentProblemSolver::JVC( mat_1, size, LAP::TSearchParam::Max, maxcost, resolution, actual );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_1_mack_max )
{
    int size = mat_1.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    LAP::CAssignmentProblemSolver::Mack( mat_1, size, LAP::TSearchParam::Max, actual );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_1_hungarian_max )
{
    int size = mat_1.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    LAP::CAssignmentProblemSolver::Hungarian( mat_1, size, LAP::TSearchParam::Max, actual );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_2_min )

BOOST_AUTO_TEST_CASE( test_mat_2_jvc_min )
{
    int size = mat_2.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double maxcost = mat_2.max();
    double resolution = 1e-6;
    LAP::CAssignmentProblemSolver::JVC( mat_2, size, LAP::TSearchParam::Min, maxcost, resolution, actual );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_2_mack_min )
{
    int size = mat_2.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    LAP::CAssignmentProblemSolver::Mack( mat_2, size, LAP::TSearchParam::Min, actual );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_2_hungarian_min )
{
    int size = mat_2.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    LAP::CAssignmentProblemSolver::Hungarian( mat_2, size, LAP::TSearchParam::Min, actual );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_2_max )

BOOST_AUTO_TEST_CASE( test_mat_2_jvc_max )
{
    int size = mat_2.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double maxcost = mat_2.max();
    double resolution = 1e-6;
    LAP::CAssignmentProblemSolver::JVC( mat_2, size, LAP::TSearchParam::Max, maxcost, resolution, actual );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_2_mack_max )
{
    int size = mat_2.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    LAP::CAssignmentProblemSolver::Mack( mat_2, size, LAP::TSearchParam::Max, actual );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_2_hungarian_max )
{
    int size = mat_2.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    LAP::CAssignmentProblemSolver::Hungarian( mat_2, size, LAP::TSearchParam::Max, actual );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE( accordance )
{
    const int n = 128;
    arma::mat p_JV( n, n, arma::fill::randn );
    arma::mat p_Mack( n, n, arma::fill::randn );
    arma::ivec actualJV = arma::ivec( n, arma::fill::zeros );
    arma::ivec actualM = arma::ivec( n, arma::fill::zeros );
    int cycle_count = 10000;
    for( int cycle = 0; cycle < cycle_count; cycle++ ) {
        std::cout << "accordance cycle = " << cycle << "/" << cycle_count << std::endl;
        arma::arma_rng::set_seed( cycle );
        p_JV.randn();
        arma::arma_rng::set_seed( cycle );
        p_Mack.randn();

        double maxcost = p_JV.max();
        double resolution = 1e-6;
        LAP::CAssignmentProblemSolver::JVC( p_JV, n, LAP::TSearchParam::Max, maxcost, resolution, actualJV );

        LAP::CAssignmentProblemSolver::Mack( p_Mack, n, LAP::TSearchParam::Max, actualM );

        for( int i = 0; i < n; i++ ) {
            BOOST_CHECK_EQUAL( ( abs( actualJV(i) - actualM(i) ) < 1e-5 ), true );
        }
    }
}

double drand( double dmin, double dmax )
{
    double d = static_cast<double>( rand() ) / RAND_MAX;
    return dmin + d * ( dmax - dmin );
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
        LAP::CAssignmentProblemSolver::JVC( assigncost, N, LAP::TSearchParam::Max, std::abs( psi_empty ), resolution, result );
    }
}
