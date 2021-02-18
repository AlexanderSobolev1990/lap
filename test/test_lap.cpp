//----------------------------------------------------------------------------------------------------------------------
///
/// \file       test_lap.cpp
/// \brief      Тестирование задач о назначениях
/// \date       14.07.20 - создан
/// \author     Соболев А.А.
///

#ifndef TEST_LAP_H
#define TEST_LAP_H

#include <QtTest>

#include <lap.h>

class test_lap : public QObject
{
    Q_OBJECT

public:
    test_lap();
    ~test_lap();

private slots:

    void test_LAP_Mack_min_1();
    void test_LAP_Mack_max_1();

    void test_LAP_Mack_min_2();
    void test_LAP_Mack_max_2();

    void test_LAP_Mack_max_3();

    void test_LAP_JVC_min_1();
    void test_LAP_JV_max_1();

    void test_LAP_JV_min_2();
    void test_LAP_JV_max_2();

    void test_LAP_JV_max_3();
    void test_LAP_JV_max_4();
    void test_LAP_JV_max_5();

    void test_LAP_Mack_speed_128_order();
    void test_LAP_JV_speed_128_order();

    void test_LAP_JV_cycling();
    void test_LAP_accordance();
};

test_lap::test_lap()
{

}

test_lap::~test_lap()
{

}

void test_lap::test_LAP_Mack_min_1()
{
    arma::mat p =
    {
        { 7.0, 2.0, 1.0, 9.0, 4.0 },
        { 9.0, 6.0, 9.0, 6.0, 5.0 },
        { 3.0, 8.0, 3.0, 1.0, 8.0 },
        { 7.0, 9.0, 4.0, 2.0, 2.0 },
        { 8.0, 4.0, 7.0, 4.0, 8.0 }
    };
    arma::mat p2 = p;
    arma::ivec expected = { 2, 4, 0, 3, 1 };
    arma::ivec actual = arma::ivec( 5, arma::fill::zeros );    
    LAP::Mack( p, 5, LAP::TSearchParam::Min, actual );
    double eps = 1e-6;
    QCOMPARE( arma::approx_equal( actual, expected, "absdiff", eps ), true );
    QBENCHMARK {
        LAP::Mack( p2, 5, LAP::TSearchParam::Min, actual );
    }
}

void test_lap::test_LAP_Mack_max_1()
{
    arma::mat p =
    {
        { 7.0, 2.0, 1.0, 9.0, 4.0 },
        { 9.0, 6.0, 9.0, 6.0, 5.0 },
        { 3.0, 8.0, 3.0, 1.0, 8.0 },
        { 7.0, 9.0, 4.0, 2.0, 2.0 },
        { 8.0, 4.0, 7.0, 4.0, 8.0 }
    };
    arma::mat p2 = p;
    arma::ivec expected = { 3,	2,	4,	1,	0 };
    arma::ivec actual = arma::ivec( 5, arma::fill::zeros );
    LAP::Mack( p, 5, LAP::TSearchParam::Max, actual );
    double eps = 1e-6;
    QCOMPARE( arma::approx_equal( actual, expected, "absdiff", eps ), true );    
    QBENCHMARK {
        LAP::Mack( p2, 5, LAP::TSearchParam::Max, actual );
    }
}

void test_lap::test_LAP_Mack_min_2()
{
    arma::mat p =
    {
        { 93.0, 93.0, 91.0, 94.0, 99.0, 99.0, 90.0, 92.0 },
        { 96.0, 93.0, 90.0, 94.0, 98.0, 96.0, 97.0, 91.0 },
        { 96.0, 90.0, 91.0, 90.0, 92.0, 90.0, 93.0, 96.0 },
        { 93.0, 94.0, 95.0, 96.0, 97.0, 10.0, 92.0, 93.0 },
        { 94.0, 93.0, 95.0, 91.0, 90.0, 97.0, 96.0, 92.0 },
        { 94.0, 93.0, 96.0, 90.0, 93.0, 89.0, 88.0, 91.0 },
        { 94.0, 96.0, 91.0, 90.0, 95.0, 93.0, 92.0, 94.0 },
        { 93.0, 94.0, 6.0,  95.0, 91.0, 99.0, 91.0, 96.0 }
    };
    arma::mat p2 = p;
    arma::ivec expected = { 0, 7, 1, 5, 4, 6, 3, 2 };
    arma::ivec actual = arma::ivec( 8, arma::fill::zeros );
    LAP::Mack( p, 8, LAP::TSearchParam::Min, actual );
    double eps = 1e-6;
    QCOMPARE( arma::approx_equal( actual, expected, "absdiff", eps ), true );    
    QBENCHMARK {
        LAP::Mack( p2, 8, LAP::TSearchParam::Min, actual );
    }
}

void test_lap::test_LAP_Mack_max_2()
{
    arma::mat p =
    {
        { 93.0, 93.0, 91.0, 94.0, 99.0, 99.0, 90.0, 92.0 },
        { 96.0, 93.0, 90.0, 94.0, 98.0, 96.0, 97.0, 91.0 },
        { 96.0, 90.0, 91.0, 90.0, 92.0, 90.0, 93.0, 96.0 },
        { 93.0, 94.0, 95.0, 96.0, 97.0, 10.0, 92.0, 93.0 },
        { 94.0, 93.0, 95.0, 91.0, 90.0, 97.0, 96.0, 92.0 },
        { 94.0, 93.0, 96.0, 90.0, 93.0, 89.0, 88.0, 91.0 },
        { 94.0, 96.0, 91.0, 90.0, 95.0, 93.0, 92.0, 94.0 },
        { 93.0, 94.0, 6.0,  95.0, 91.0, 99.0, 91.0, 96.0 }
    };
    arma::mat p2 = p;
    arma::ivec expected = { 4, 0, 7, 3, 6, 2, 1, 5 };
    arma::ivec actual = arma::ivec( 8, arma::fill::zeros );
    LAP::Mack( p, 8, LAP::TSearchParam::Max, actual );
    double eps = 1e-6;
    QCOMPARE( arma::approx_equal( actual, expected, "absdiff", eps ), true );    
    QBENCHMARK {
        LAP::Mack( p2, 8, LAP::TSearchParam::Max, actual );
    }
}

void test_lap::test_LAP_Mack_max_3()
{
    arma::mat p =
    {
        { 1.0, 2.0, 0.0, 0.0 },
        { 0.0, 3.0, 0.0, 4.0 },
        { 5.0, 0.0, 0.0, 6.0 },
        { 0.0, 0.0, 0.0, 0.0 }
    };
    arma::mat p2 = p;
    arma::ivec expected = { 1, 3, 0, 2 };
    arma::ivec actual = arma::ivec( 4, arma::fill::zeros );
    LAP::Mack( p, 4, LAP::TSearchParam::Max, actual );
    double eps = 1e-6;
    QCOMPARE( arma::approx_equal( actual, expected, "absdiff", eps ), true );    
    QBENCHMARK {
        LAP::Mack( p2, 4, LAP::TSearchParam::Max, actual );
    }
}

void test_lap::test_LAP_JVC_min_1()
{
    arma::mat p
    {
        { 7.0, 2.0, 1.0, 9.0, 4.0 },
        { 9.0, 6.0, 9.0, 6.0, 5.0 },
        { 3.0, 8.0, 3.0, 1.0, 8.0 },
        { 7.0, 9.0, 4.0, 2.0, 2.0 },
        { 8.0, 4.0, 7.0, 4.0, 8.0 }
    };    
    arma::ivec expected = { 2, 4, 0, 3, 1 };
    arma::ivec actual = arma::ivec( 5, arma::fill::zeros );
    double maxcost = p.max();
    double resolution = 1e-6;
    LAP::JVC( p, 5, LAP::TSearchParam::Min, maxcost, resolution, actual );
    double eps = 1e-6;
    QCOMPARE( arma::approx_equal( actual, expected, "absdiff", eps ), true );    
    QBENCHMARK {
        LAP::JVC( p, 5, LAP::TSearchParam::Min, maxcost, resolution, actual );
    }
}

void test_lap::test_LAP_JV_max_1()
{
    arma::mat p
    {
        { 7.0, 2.0, 1.0, 9.0, 4.0 },
        { 9.0, 6.0, 9.0, 6.0, 5.0 },
        { 3.0, 8.0, 3.0, 1.0, 8.0 },
        { 7.0, 9.0, 4.0, 2.0, 2.0 },
        { 8.0, 4.0, 7.0, 4.0, 8.0 }
    };    
    arma::ivec expected = { 3,	2,	4,	1,	0 };
    arma::ivec actual = arma::ivec( 5, arma::fill::zeros );
    double maxcost = p.max();
    double resolution = 1e-6;
    LAP::JVC( p, 5, LAP::TSearchParam::Max, maxcost, resolution, actual );
    double eps = 1e-6;
    QCOMPARE( arma::approx_equal( actual, expected, "absdiff", eps ), true );    
    QBENCHMARK {        
        LAP::JVC( p, 5, LAP::TSearchParam::Max, maxcost, resolution, actual );
    }
}

void test_lap::test_LAP_JV_min_2()
{
    arma::mat p
    {
        { 93.0, 93.0, 91.0, 94.0, 99.0, 99.0, 90.0, 92.0 },
        { 96.0, 93.0, 90.0, 94.0, 98.0, 96.0, 97.0, 91.0 },
        { 96.0, 90.0, 91.0, 90.0, 92.0, 90.0, 93.0, 96.0 },
        { 93.0, 94.0, 95.0, 96.0, 97.0, 10.0, 92.0, 93.0 },
        { 94.0, 93.0, 95.0, 91.0, 90.0, 97.0, 96.0, 92.0 },
        { 94.0, 93.0, 96.0, 90.0, 93.0, 89.0, 88.0, 91.0 },
        { 94.0, 96.0, 91.0, 90.0, 95.0, 93.0, 92.0, 94.0 },
        { 93.0, 94.0, 6.0,  95.0, 91.0, 99.0, 91.0, 96.0 }
    };    
    arma::ivec expected = { 0, 7, 1, 5, 4, 6, 3, 2 };
    arma::ivec actual = arma::ivec( 8, arma::fill::zeros );
    double maxcost = p.max();
    double resolution = 1e-6;
    LAP::JVC( p, 8, LAP::TSearchParam::Min, maxcost, resolution, actual );
    double eps = 1e-6;
    QCOMPARE( arma::approx_equal( actual, expected, "absdiff", eps ), true );    
    QBENCHMARK {        
        LAP::JVC( p, 8, LAP::TSearchParam::Min, maxcost, resolution, actual );
    }
}

void test_lap::test_LAP_JV_max_2()
{
    arma::mat p
    {
        { 93.0, 93.0, 91.0, 94.0, 99.0, 99.0, 90.0, 92.0 },
        { 96.0, 93.0, 90.0, 94.0, 98.0, 96.0, 97.0, 91.0 },
        { 96.0, 90.0, 91.0, 90.0, 92.0, 90.0, 93.0, 96.0 },
        { 93.0, 94.0, 95.0, 96.0, 97.0, 10.0, 92.0, 93.0 },
        { 94.0, 93.0, 95.0, 91.0, 90.0, 97.0, 96.0, 92.0 },
        { 94.0, 93.0, 96.0, 90.0, 93.0, 89.0, 88.0, 91.0 },
        { 94.0, 96.0, 91.0, 90.0, 95.0, 93.0, 92.0, 94.0 },
        { 93.0, 94.0, 6.0,  95.0, 91.0, 99.0, 91.0, 96.0 }
    };    
    arma::ivec expected = { 4, 0, 7, 3, 6, 2, 1, 5 };
    arma::ivec actual = arma::ivec( 8, arma::fill::zeros );
    double maxcost = p.max();
    double resolution = 1e-6;
    LAP::JVC( p, 8, LAP::TSearchParam::Max, maxcost, resolution, actual );
    double eps = 1e-6;
    QCOMPARE( arma::approx_equal( actual, expected, "absdiff", eps ), true );    
    QBENCHMARK {        
        LAP::JVC( p, 8, LAP::TSearchParam::Max, maxcost, resolution, actual );
    }
}

void test_lap::test_LAP_JV_max_3()
{
    arma::mat p =
    {
        { 1.0, 2.0, 0.0, 0.0 },
        { 0.0, 3.0, 0.0, 4.0 },
        { 5.0, 0.0, 0.0, 6.0 },
        { 0.0, 0.0, 0.0, 0.0 }
    };
    arma::ivec expected = { 1, 3, 0, 2 };
    arma::ivec actual = arma::ivec( 4, arma::fill::zeros );
    double maxcost = p.max();
    double resolution = 1e-6;
    LAP::JVC( p, 4, LAP::TSearchParam::Max, maxcost, resolution, actual );
    double eps = 1e-6;
    QCOMPARE( arma::approx_equal( actual, expected, "absdiff", eps ), true );
    QBENCHMARK {
        LAP::JVC( p, 4, LAP::TSearchParam::Max, maxcost, resolution, actual );
    }
}

void test_lap::test_LAP_JV_max_4()
{
    arma::mat p
    {
        { -9.762, -1.0e6, -1.0e6 },
        { -9.758, -1.0e6, -1.0e6 },
        { -1.0e6, -1.0e6, -1.0e6 },
    };
    arma::ivec expected = { 2, 0, 1 };
    arma::ivec actual = arma::ivec( 3, arma::fill::zeros );
    double maxcost = p.max();
    double resolution = 1e-6;
    LAP::JVC( p, 3, LAP::TSearchParam::Max, maxcost, resolution, actual );
    double eps = 1e-6;
    QCOMPARE( arma::approx_equal( actual, expected, "absdiff", eps ), true );
    QBENCHMARK {
        double maxcost = p.max();
        double resolution = 1e-6;
        LAP::JVC( p, 3, LAP::TSearchParam::Max, maxcost, resolution, actual );
    }
}

void test_lap::test_LAP_JV_max_5()
{
    arma::mat p // Матрица ценности привязок
    {
        { 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 33.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 5.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0 },
        { 40.0,0.0, 600.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
    };
    arma::ivec expected = { 3, 1, 0, 2 };
    arma::ivec actual = arma::ivec( 8, arma::fill::zeros );
    double maxcost = p.max();
    double resolution = 1e-6;
    LAP::JVC( p, 8, LAP::TSearchParam::Max, maxcost, resolution, actual );
    // Сравниваем только 4 элемента, потому что остальные нули
    for( int i = 0; i < 4; i++ ) {
        QCOMPARE( actual[i], expected[i] );
    }
    QBENCHMARK {
        LAP::JVC( p, 8, LAP::TSearchParam::Max, maxcost, resolution, actual );
    }
}

void test_lap::test_LAP_Mack_speed_128_order()
{
    arma::arma_rng::set_seed( 1 );
    arma::mat p( 128, 128, arma::fill::randu );
    arma::ivec actualMack = arma::ivec( 128, arma::fill::zeros );
    QBENCHMARK {
        LAP::Mack( p, 128, LAP::TSearchParam::Max, actualMack );
    }
}

void test_lap::test_LAP_JV_speed_128_order()
{
    arma::arma_rng::set_seed( 1 );
    arma::mat p( 128, 128, arma::fill::randu );
    arma::ivec actualJonkerVolgenant = arma::ivec( 128, arma::fill::zeros );
    double maxcost = p.max();
    double resolution = 1e-6;
    QBENCHMARK {        
        LAP::JVC( p, 128, LAP::TSearchParam::Max, maxcost, resolution, actualJonkerVolgenant );
    }
}

double drand( double dmin, double dmax )
{
    double d = static_cast<double>( rand() ) / RAND_MAX;
    return dmin + d * ( dmax - dmin );
}

void test_lap::test_LAP_JV_cycling()
{
    // Тест проверки на зацикливание метода
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
        std::cout << "test_LAP_JV_cycling cycle = " << cycle << "/" << cycle_count << std::endl;
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
        LAP::JVC( assigncost, N, LAP::TSearchParam::Max, std::abs( psi_empty ), resolution, result );
    }        
}

void test_lap::test_LAP_accordance()
{
    const int n = 128;
    arma::mat p_JV( n, n, arma::fill::randn );
    arma::mat p_Mack( n, n, arma::fill::randn );
    arma::ivec actualJV = arma::ivec( n, arma::fill::zeros );
    arma::ivec actualM = arma::ivec( n, arma::fill::zeros );
    int cycle_count = 1000;
    for( int cycle = 0; cycle < cycle_count; cycle++ ) {
        std::cout << "test_LAP_accordance cycle = " << cycle << "/" << cycle_count << std::endl;
        arma::arma_rng::set_seed( cycle );
        p_JV.randn();        
        arma::arma_rng::set_seed( cycle );
        p_Mack.randn();

        double maxcost = p_JV.max();
        double resolution = 1e-6;
        LAP::JVC( p_JV, n, LAP::TSearchParam::Max, maxcost, resolution, actualJV );

        LAP::Mack( p_Mack, n, LAP::TSearchParam::Max, actualM );

        for( int i = 0; i < n; i++ ) {
            QCOMPARE( ( abs( actualJV(i) - actualM(i) ) < 1e-5 ), true );
        }
    }
}

QTEST_APPLESS_MAIN(test_lap)

#include "test_lap.moc"

#endif
