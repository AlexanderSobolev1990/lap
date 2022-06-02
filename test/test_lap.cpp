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

#include <boost/test/unit_test.hpp>
#include <vector>
#include <map>
#include <armadillo>
#include <random>

#include <timing.h>
#include <matplotlibcpp.h>
#include <lap.h>
#include <sparse.h>

BOOST_AUTO_TEST_CASE( denseSize5to50 )
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
    plt::title( "Время решения задачи о назначениях на плотных матрицах" );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, [мс]" );

    std::vector<int> dimensionXlim = { 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 };
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

    for( auto &n : dimensionXlim ) {

        arma::mat mat_JVCdense( n, n, arma::fill::randn );
        arma::mat mat_Mack( n, n, arma::fill::randn );
        arma::mat mat_Hungarian( n, n, arma::fill::randn );
        arma::ivec actualJVC = arma::ivec( n, arma::fill::zeros );
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
            mat_JVCdense.randn();
            arma::arma_rng::set_seed( cycle );
            mat_Mack.randn();
            arma::arma_rng::set_seed( cycle );
            mat_Hungarian.randn();

            double maxcost = mat_JVCdense.max();
            double resolution = 1e-6;

            timer.at( "JVCdense" ).StartTimer();
            SPML::LAP::JVCdense( mat_JVCdense, n, SPML::LAP::TSearchParam::SP_Max, maxcost, resolution, actualJVC, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

            timer.at( "Mack" ).StartTimer();
            SPML::LAP::Mack( mat_Mack, n, SPML::LAP::TSearchParam::SP_Max, maxcost, resolution, actualMack, lapcostMack );
            timer.at( "Mack" ).EndTimer();

            BOOST_CHECK_EQUAL( std::abs( lapcostJVCdense - lapcostMack ) < 1e-5, true );

            if( n <= boundN ) {
                timer.at( "Hungarian" ).StartTimer();
                SPML::LAP::Hungarian( mat_Hungarian, n, SPML::LAP::TSearchParam::SP_Max, maxcost, resolution, actualHungarian, lapcostHungarian );
                timer.at( "Hungarian" ).EndTimer();

                BOOST_CHECK_EQUAL( std::abs( lapcostJVCdense - lapcostHungarian ) < 1e-5, true );
            }

            for( int i = 0; i < n; i++ ) {
                BOOST_CHECK_EQUAL( ( std::abs( actualJVC(i) - actualMack(i) ) < 1e-5 ), true );
                if( n <= boundN ) {
                    BOOST_CHECK_EQUAL( ( std::abs( actualJVC(i) - actualHungarian(i) ) < 1e-5 ), true );
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
    plt::save( "time_2d_assignment_compare_dense_N_from_2_to_50.png", dpi );
    plt::show();
    plt::close();
    plt::clf();
    plt::cla();
}

BOOST_AUTO_TEST_CASE( denseSize50to1000 )
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
    plt::title( "Время решения задачи о назначениях на плотных матрицах" );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, [мс]" );

    std::vector<int> dimensionXlim =     { 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000 };
    std::vector<double> dimensionLong =  { 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000 };
    std::vector<double> dimensionShort = { 50, 100, 150, 200, 250, 300, 350, 400, 450, 500 };

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

    for( auto &n : dimensionXlim ) {

        arma::mat mat_JVCdense( n, n, arma::fill::randn );
        arma::mat mat_Mack( n, n, arma::fill::randn );
        arma::mat mat_Hungarian( n, n, arma::fill::randn );
        arma::ivec actualJVC = arma::ivec( n, arma::fill::zeros );
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
            mat_JVCdense.randn();
            arma::arma_rng::set_seed( cycle );
            mat_Mack.randn();
            arma::arma_rng::set_seed( cycle );
            mat_Hungarian.randn();

            double maxcost = mat_JVCdense.max();
            double resolution = 1e-6;

            timer.at( "JVCdense" ).StartTimer();
            SPML::LAP::JVCdense( mat_JVCdense, n, SPML::LAP::TSearchParam::SP_Max, maxcost, resolution, actualJVC, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

            timer.at( "Mack" ).StartTimer();
            SPML::LAP::Mack( mat_Mack, n, SPML::LAP::TSearchParam::SP_Max, maxcost, resolution, actualMack, lapcostMack );
            timer.at( "Mack" ).EndTimer();

            BOOST_CHECK_EQUAL( std::abs( lapcostJVCdense - lapcostMack ) < 1e-5, true );

            if( n <= boundN ) {
                timer.at( "Hungarian" ).StartTimer();
                SPML::LAP::Hungarian( mat_Hungarian, n, SPML::LAP::TSearchParam::SP_Max, maxcost, resolution, actualHungarian, lapcostHungarian );
                timer.at( "Hungarian" ).EndTimer();

                BOOST_CHECK_EQUAL( std::abs( lapcostJVCdense - lapcostHungarian ) < 1e-5, true );
            }

            for( int i = 0; i < n; i++ ) {
                BOOST_CHECK_EQUAL( ( abs( actualJVC(i) - actualMack(i) ) < 1e-5 ), true );
                if( n <= boundN ) {
                    BOOST_CHECK_EQUAL( ( abs( actualJVC(i) - actualHungarian(i) ) < 1e-5 ), true );
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
    plt::save( "time_2d_assignment_compare_dense_N_from_5_to_1000.png", dpi );
    plt::show();
    plt::close();
    plt::clf();
    plt::cla();
//    Py_Finalize();
}

BOOST_AUTO_TEST_CASE( sparseSize5to50 )
{
    SPML::LAP::TSearchParam sp = SPML::LAP::TSearchParam::SP_Max;
    double maxvalue = 1.0e6;
    double resolution = 1.0e-6;

    bool print = true; //false; //

    int cycle_count = 1000; //5000

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
    plt::title( "Время решения задачи о назначениях на разреженных матрицах" );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, [мс]" );

    std::vector<int> dimensionXlim = { 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 };
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

    std::mt19937 generator; // Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_double_0_1( 0.0, 1.0 ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности
    double threshold = 0.2; // Степень заполненности/разрежености матрицы

    for( auto &n : dimensionXlim ) {
        std::uniform_int_distribution<int> random_uint_0_n( 0, ( n - 1 ) );

        SPML::Sparse::CMatrixCSR<double> mat_JVCsparse;
        arma::mat mat_JVCdense( n, n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse( n, n, arma::fill::zeros );
        arma::mat mat_Mack( n, n, arma::fill::zeros );
        arma::mat mat_Hungarian( n, n, arma::fill::zeros );
        arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualJVCsparse = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );
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
            arma::arma_rng::set_seed( cycle );
            mat_JVCdense.randu(); // 0..1
            mat_JVCdenseForSparse = mat_JVCdense;
/*
            for( int i = 0; i < n; i++ ) {
                int zeros_in_row = 0;
                for( int j = 0; j < n; j++ ) {
                    double randomDouble = random_double_0_1( generator ); // Случайное вещественное число от 0 до 1
                    if( randomDouble > threshold ) { // Проредим матрицу
                        mat_JVCdense(i, j) = maxvalue;
                        mat_JVCdenseForSparse(i, j) = 0.0;
                        zeros_in_row++;
                    }
                }
                if( zeros_in_row == n ) { // Слишком проредили
                    int randomInt = random_uint_0_n( generator );
                    double randomDouble = random_double_0_1( generator ); // Случайное вещественное число от 0 до 1
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
                    double randomDouble = random_double_0_1( generator ); // Случайное вещественное число от 0 до 1
                    mat_JVCdense(randomInt, j) = randomDouble;
                    mat_JVCdenseForSparse(randomInt, j) = randomDouble;
                }
            }
            */
//            mat_JVCdense *= 1000.0;
//            mat_JVCdenseForSparse *= 1000.0;

            double maxcost = maxvalue;//mat_JVCdenseForSparse.max() + 1.0;

//            mat_JVCdense.print();
//            mat_JVCdenseForSparse.print();

            SPML::Sparse::MatrixDenseToCSR( mat_JVCdenseForSparse, mat_JVCsparse );
            mat_Mack = mat_JVCdense;
            mat_Hungarian = mat_JVCdense;

            timer.at( "JVCdense" ).StartTimer();
            SPML::LAP::JVCdense( mat_JVCdense, n, sp, maxcost, resolution, actualJVCdense, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

            timer.at( "JVCsparse" ).StartTimer();
            SPML::LAP::JVCsparseNEW( mat_JVCsparse.csr_val, mat_JVCsparse.csr_first, mat_JVCsparse.csr_kk,
                sp, std::abs( maxcost ), resolution, actualJVCsparse, lapcostJVCsparse );
            timer.at( "JVCsparse" ).EndTimer();

            timer.at( "Mack" ).StartTimer();
            SPML::LAP::Mack( mat_Mack, n, sp, maxcost, resolution, actualMack, lapcostMack );
            timer.at( "Mack" ).EndTimer();

            BOOST_CHECK_EQUAL( std::abs( lapcostJVCdense - lapcostMack ) < 1e-6, true );

            if( n <= boundN ) {
                timer.at( "Hungarian" ).StartTimer();
                SPML::LAP::Hungarian( mat_Hungarian, n, sp, maxcost, resolution, actualHungarian, lapcostHungarian );
                timer.at( "Hungarian" ).EndTimer();

                BOOST_CHECK_EQUAL( std::abs( lapcostJVCdense - lapcostHungarian ) < 1e-6, true );
            }

            double eps = 1e-6;
            bool eq_actualJVCdense_actualMack = arma::approx_equal( actualJVCdense, actualMack, "absdiff", eps );
            bool eq_actualJVCdense_actualJVCsparse = arma::approx_equal( actualJVCdense, actualJVCsparse, "absdiff", eps );
            bool eq_actualJVCdense_actualHungarian = false;
            if( n <= boundN ) {
                eq_actualJVCdense_actualHungarian = arma::approx_equal( actualJVCdense, actualHungarian, "absdiff", eps );
            }
            if( !eq_actualJVCdense_actualMack ||
                !eq_actualJVCdense_actualJVCsparse ||
                ( ( n <= boundN ) && ( !eq_actualJVCdense_actualHungarian ) ) )
            {
                int abc = 0;
            }

            // Сравнивать точные вхождения при больших матрицах немного некорректно........
//            BOOST_CHECK_EQUAL( eq_actualJVCdense_actualMack, true );
//            BOOST_CHECK_EQUAL( eq_actualJVCdense_actualJVCsparse, true );
//            if( n <= boundN ) {
//                BOOST_CHECK_EQUAL( eq_actualJVCdense_actualHungarian, true );
//            }
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
    plt::save( "time_2d_assignment_compare_sparse_N_from_2_to_50.png", dpi );
    plt::show();
    plt::close();
    plt::clf();
    plt::cla();
}

BOOST_AUTO_TEST_CASE( sparseSize5to50_2 )
{
    SPML::LAP::TSearchParam sp = SPML::LAP::TSearchParam::SP_Max;
    double maxvalue = 1.0e6;
    double resolution = 1.0e-6;

    bool print = true; //false; //

    int cycle_count = 1000; //5000
#ifdef MATPLOTLIB
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
    plt::title( "Время решения задачи о назначениях на разреженных матрицах" );
    plt::xlabel( "Размерность задачи N" );
    plt::ylabel( "Время, [мс]" );
#endif

//    std::vector<int> dimensionXlim = { 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 };
    std::vector<int> dimensionXlim = { 5, 7, 8, 10 };//, 15, 20, 25, 30, 35, 40, 45, 50 };
    std::vector<double> dimensionLong( dimensionXlim.begin(), dimensionXlim.end() );
    std::vector<double> dimensionShort( dimensionXlim.begin(), dimensionXlim.end() );

//    std::vector<int> dimensionXlim =     { 50, 100, 200, 300, 350, 400, 500 };
//    std::vector<double> dimensionLong =  { 50, 100, 200, 300, 350, 400, 500 };
//    std::vector<double> dimensionShort = { 50, 100, 200 };

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

    std::mt19937 generator; // Генератор псевдослучайных чисел Mersenne Twister
    std::uniform_real_distribution<double> random_double_0_1( 0.0, 1.0 ); // Вещественное случайное число от 0 до 1 с равномерной плотностью вероятности
    double threshold = 0.2; // Степень заполненности/разрежености матрицы

    int counter_assign = 0;
    int counter_lapcost = 0;

    for( auto &n : dimensionXlim ) {
        std::uniform_int_distribution<int> random_uint_0_n( 0, ( n - 1 ) );

        SPML::Sparse::CMatrixCSR<double> mat_JVCsparse;
        arma::mat mat_JVCdense( n, n, arma::fill::zeros );
        arma::mat mat_JVCdenseForSparse( n, n, arma::fill::zeros );
        arma::mat mat_Mack( n, n, arma::fill::zeros );
        arma::mat mat_Hungarian( n, n, arma::fill::zeros );
        arma::ivec actualJVCdense = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualJVCsparse = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualMack = arma::ivec( n, arma::fill::zeros );
        arma::ivec actualHungarian = arma::ivec( n, arma::fill::zeros );
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
            arma::arma_rng::set_seed( cycle );
            mat_JVCdense.randu(); // 0..1
            mat_JVCdenseForSparse = mat_JVCdense;

//            bool doSparse = false;
            bool doSparse = true;

            if( doSparse ) {
                for( int i = 0; i < n; i++ ) {
                    int zeros_in_row = 0;
                    for( int j = 0; j < n; j++ ) {
                        double randomDouble = random_double_0_1( generator ); // Случайное вещественное число от 0 до 1
                        if( randomDouble > threshold ) { // Проредим матрицу
                            mat_JVCdense(i, j) = maxvalue;
                            mat_JVCdenseForSparse(i, j) = 0.0;
                            zeros_in_row++;
                        }
                    }
                    if( zeros_in_row == n ) { // Слишком проредили
                        int randomInt = random_uint_0_n( generator );
                        double randomDouble = random_double_0_1( generator ); // Случайное вещественное число от 0 до 1
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
                        double randomDouble = random_double_0_1( generator ); // Случайное вещественное число от 0 до 1
                        mat_JVCdense(randomInt, j) = randomDouble;
                        mat_JVCdenseForSparse(randomInt, j) = randomDouble;
                    }
                }
            }
//            mat_JVCdense *= 1000.0;
//            mat_JVCdenseForSparse *= 1000.0;

            SPML::Sparse::MatrixDenseToCSR( mat_JVCdenseForSparse, mat_JVCsparse );
            double maxcost = 1.0e100;
            for( int t = 0; t < mat_JVCsparse.csr_val.size(); t++ ) { // ищем минимум!
                if( mat_JVCsparse.csr_val[t] < maxcost ) {
                    maxcost = mat_JVCsparse.csr_val[t];
                }
            }
//            maxcost -= 1.0;
//            maxcost = maxvalue;//mat_JVCdenseForSparse.max() + 1.0;
            maxcost = 1.0e6;

            for( int i = 0; i < n; i++ ) {
                for( int j = 0; j < n; j++ ) {
                    if( SPML::Compare::AreEqualAbs( mat_JVCdense(i, j), maxvalue ) ) {
                        mat_JVCdense(i, j) = -maxcost;
                    }
                }
            }

            arma::mat mat_JVCdenseForSparseX2 = arma::mat( n * 2, n * 2, arma::fill::zeros );
            mat_JVCdenseForSparseX2.fill( -maxcost );
            for( int i = 0; i < n; i++ ) {
                for( int j = 0; j < n; j++ ) {
                    mat_JVCdenseForSparseX2(i, j) = mat_JVCdenseForSparse(i, j);
                }
            }
            SPML::Sparse::CMatrixCSR<double> mat_JVCsparseX2;
            SPML::Sparse::MatrixDenseToCSR( mat_JVCdenseForSparseX2, mat_JVCsparseX2 );

            //////////////////////////////
//            mat_JVCdense.print();
//            mat_JVCdenseForSparse.print();
//            mat_JVCdenseForSparseX2.print();
            //////////////////////////////

            mat_Mack = mat_JVCdense;
            mat_Hungarian = mat_JVCdense;

            timer.at( "JVCdense" ).StartTimer();
            SPML::LAP::JVCdense( mat_JVCdense, n, sp, maxcost, resolution, actualJVCdense, lapcostJVCdense );
            timer.at( "JVCdense" ).EndTimer();

            timer.at( "JVCsparse" ).StartTimer();
//            SPML::LAP::JVCsparse( mat_JVCsparse, n, sp, 1.0e6, resolution, actualJVCsparse, lapcostJVCsparse, SPML::LAP::TFindPath::FP_1 );
//            SPML::LAP::JVCsparse( mat_JVCsparse, n, sp, maxcost, resolution, actualJVCsparse, lapcostJVCsparse, SPML::LAP::TFindPath::FP_1 );
//            SPML::LAP::JVCsparse( mat_JVCsparse, n, sp, -maxcost, 1.0e-5, actualJVCsparse, lapcostJVCsparse, SPML::LAP::TFindPath::FP_1 );

//            int resSparse = SPML::LAP::JVCsparseNEW( mat_JVCsparse.csr_val, mat_JVCsparse.csr_first, mat_JVCsparse.csr_kk, sp,
//                maxcost, resolution, actualJVCsparse, lapcostJVCsparse );

//            int resSparse = SPML::LAP::JVCsparseNEW( mat_JVCsparseX2.csr_val, mat_JVCsparseX2.csr_first, mat_JVCsparseX2.csr_kk,
//                sp, maxcost, resolution, actualJVCsparse, lapcostJVCsparse );
            timer.at( "JVCsparse" ).EndTimer();

            timer.at( "Mack" ).StartTimer();
            SPML::LAP::Mack( mat_Mack, n, sp, maxcost, resolution, actualMack, lapcostMack );
            timer.at( "Mack" ).EndTimer();

            if( n <= boundN ) {
                timer.at( "Hungarian" ).StartTimer();
                SPML::LAP::Hungarian( mat_Hungarian, n, sp, maxcost, resolution, actualHungarian, lapcostHungarian );
                timer.at( "Hungarian" ).EndTimer();
            }

            bool eq_lapcostJVCdense_lapcostMack = std::abs( lapcostJVCdense - lapcostMack ) < 1e-5;
            bool eq_lapcostJVCdense_lapcostJVCsparse = std::abs( lapcostJVCdense - lapcostJVCsparse ) < 1e-5;
            bool eq_lapcostJVCdense_lapcostHungarian = false;

            if( n <= boundN ) {
                eq_lapcostJVCdense_lapcostHungarian = std::abs( lapcostJVCdense - lapcostHungarian ) < 1e-5;
            }

            if( !eq_lapcostJVCdense_lapcostMack ||
                !eq_lapcostJVCdense_lapcostJVCsparse ||
                ( ( n <= boundN ) && ( !eq_lapcostJVCdense_lapcostHungarian ) ) )
            {
                counter_lapcost++;
            }

//            BOOST_CHECK_EQUAL( eq_lapcostJVCdense_lapcostMack, true );
//            BOOST_CHECK_EQUAL( std::abs( lapcostJVCdense - lapcostJVCsparse ) < 1e-5, true );
//            if( n <= boundN ) {
//                BOOST_CHECK_EQUAL( std::abs( lapcostJVCdense - lapcostHungarian ) < 1e-5, true );
//            }

            double eps = 1e-6;
            bool eq_actualJVCdense_actualMack = arma::approx_equal( actualJVCdense, actualMack, "absdiff", eps );
            bool eq_actualJVCdense_actualJVCsparse = arma::approx_equal( actualJVCdense, actualJVCsparse, "absdiff", eps );
            bool eq_actualJVCdense_actualHungarian = false;
            if( n <= boundN ) {
                eq_actualJVCdense_actualHungarian = arma::approx_equal( actualJVCdense, actualHungarian, "absdiff", eps );
            }
            if( !eq_actualJVCdense_actualMack ||
                !eq_actualJVCdense_actualJVCsparse ||
                ( ( n <= boundN ) && ( !eq_actualJVCdense_actualHungarian ) ) )
            {
                counter_assign++;
            }

            // Сравнивать точные вхождения при больших матрицах немного некорректно........
//            BOOST_CHECK_EQUAL( eq_actualJVCdense_actualMack, true );
//            BOOST_CHECK_EQUAL( eq_actualJVCdense_actualJVCsparse, true );
//            if( n <= boundN ) {
//                BOOST_CHECK_EQUAL( eq_actualJVCdense_actualHungarian, true );
//            }
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
    plt::save( "time_2d_assignment_compare_sparse_N_from_2_to_50.png", dpi );
    plt::show();
    plt::close();
    plt::clf();
    plt::cla();
#endif
}


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
//----------------------------------------------------------------------------------------------------------------------
// 3
arma::mat mat_3_dense = {
    { 1.0, 2.0, 0.0, 0.0, 0.0, 0.0 },
    { 0.0, 3.0, 0.0, 4.0, 0.0, 0.0 },
    { 0.0, 0.0, 5.0, 6.0, 7.0, 0.0 },
    { 8.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    { 0.0, 0.0, 0.0, 0.0, 9.0, 0.0 },
    { 0.0, 0.0, 0.0, 0.0, 0.0,10.0 },
};
arma::ivec expected_3_max = { 1, 3, 2, 0, 4, 5 };
//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_1_min )

BOOST_AUTO_TEST_CASE( test_mat_1_jvc_dense_min )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double maxcost = mat_1_dense.max();
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::JVCdense( mat_1_dense, size, SPML::LAP::TSearchParam::SP_Min, maxcost, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_1_mack_min )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double maxcost = mat_1_dense.max();
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::Mack( mat_1_dense, size, SPML::LAP::TSearchParam::SP_Min, maxcost, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_1_hungarian_min )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double maxcost = mat_1_dense.max();
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::Hungarian( mat_1_dense, size, SPML::LAP::TSearchParam::SP_Min, maxcost, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_1_max )

BOOST_AUTO_TEST_CASE( test_mat_1_jvc_dense_max )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double maxcost = mat_1_dense.max();
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::JVCdense( mat_1_dense, size, SPML::LAP::TSearchParam::SP_Max, maxcost, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_1_mack_max )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double maxcost = mat_1_dense.max();
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::Mack( mat_1_dense, size, SPML::LAP::TSearchParam::SP_Max, maxcost, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_1_hungarian_max )
{
    int size = mat_1_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double maxcost = mat_1_dense.max();
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::Hungarian( mat_1_dense, size, SPML::LAP::TSearchParam::SP_Max, maxcost, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_1_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_2_min )

BOOST_AUTO_TEST_CASE( test_mat_2_jvc_dense_min )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double maxcost = mat_2_dense.max();
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::JVCdense( mat_2_dense, size, SPML::LAP::TSearchParam::SP_Min, maxcost, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_2_mack_min )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double maxcost = mat_1_dense.max();
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::Mack( mat_2_dense, size, SPML::LAP::TSearchParam::SP_Min, maxcost, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_2_hungarian_min )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double maxcost = mat_1_dense.max();
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::Hungarian( mat_2_dense, size, SPML::LAP::TSearchParam::SP_Min, maxcost, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_min, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_2_max )

BOOST_AUTO_TEST_CASE( test_mat_2_jvc_dense_max )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double maxcost = mat_2_dense.max();
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::JVCdense( mat_2_dense, size, SPML::LAP::TSearchParam::SP_Max, maxcost, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_2_mack_max )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double maxcost = mat_1_dense.max();
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::Mack( mat_2_dense, size, SPML::LAP::TSearchParam::SP_Max, maxcost, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( test_mat_2_hungarian_max )
{
    int size = mat_2_dense.n_cols;
    arma::ivec actual = arma::ivec( size, arma::fill::zeros );
    double maxcost = mat_1_dense.max();
    double resolution = 1e-6;
    double lapcost;
    SPML::LAP::Hungarian( mat_2_dense, size, SPML::LAP::TSearchParam::SP_Max, maxcost, resolution, actual, lapcost );
    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( actual, expected_2_max, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_SUITE_END()

//----------------------------------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE( test_mat_3_min )

BOOST_AUTO_TEST_CASE( test_mat_3_jvc_sparse_min )
{
    std::vector<double> csr_val;
    std::vector<int> csr_first;
    std::vector<int> csr_kk;
    SPML::Sparse::MatrixDenseToCSR( mat_3_dense, csr_val, csr_first, csr_kk );

    unsigned size = mat_3_dense.n_cols;
    arma::ivec actual1 = arma::ivec( size, arma::fill::zeros );
    arma::ivec actual2 = arma::ivec( size, arma::fill::zeros );
    double maxcost = mat_1_dense.max();
    double resolution = 1e-6;
    double lapcost1, lapcost2;
    SPML::LAP::JVCdense( mat_3_dense, size, SPML::LAP::TSearchParam::SP_Max, maxcost, resolution, actual1, lapcost1 );
    SPML::LAP::JVCsparse( csr_val, csr_kk, csr_first, size, SPML::LAP::TSearchParam::SP_Max, maxcost, resolution, actual2, lapcost2 );

    double eps = 1e-6;
    BOOST_CHECK_EQUAL( ( std::abs( lapcost1 - lapcost2 ) < eps ), true );
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

        double maxcost = mat_JVCdense.max();// + 1.0;
        double resolution = 1e-6;
        SPML::LAP::JVCdense( mat_JVCdense, n, SPML::LAP::TSearchParam::SP_Max, maxcost, resolution, actualJVCdense, lapcostJVCdense );

        SPML::LAP::Mack( mat_Mack, n, SPML::LAP::TSearchParam::SP_Max, maxcost, resolution, actualMack, lapcostMack );

        BOOST_CHECK_EQUAL( lapcostJVCdense, lapcostMack );

        for( int i = 0; i < n; i++ ) {
            BOOST_CHECK_EQUAL( ( std::abs( actualJVCdense(i) - actualMack(i) ) < 1e-5 ), true );
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
