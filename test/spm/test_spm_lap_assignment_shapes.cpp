//----------------------------------------------------------------------------------------------------------------------
///
/// \file       test_spm_lap_assignment_shapes.cpp
/// \brief      Проверка JVC и Murty на матрицах всех размеров от 1x1 до 10x10
///

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

#include <armadillo>
#include <boost/test/unit_test.hpp>

#include "spm/lap/jvcdense.hpp"
#include "spm/lap/jvcsparse.hpp"
#include "spm/lap/murty.hpp"
#include "spm/lap/sparse.hpp"

namespace {

const double emptyEdge = 1e30;
const double precision = 1e-9;
const double checkPrecision = 1e-8;
const std::size_t kBest = 24;

struct ExpectedAssignments
{
    std::uint64_t count = 0;
    std::vector<double> costs;
};

unsigned bitCount( unsigned value )
{
    unsigned count = 0;
    for( ; value != 0; value >>= 1 ) {
        count += value & 1u;
    }
    return count;
}

// Независимый эталон: состояние задаётся множеством занятых столбцов, строки обрабатываются по порядку.
// Для одного множества будущие продолжения одинаковы, поэтому достаточно хранить K лучших частичных сумм.
// Кратность одинаковых сумм сохраняется: это разные назначения. Отдельно считаем все допустимые назначения.
ExpectedAssignments expectedAssignments( const arma::mat &cost, SPM::LAP::TSearchParam sp,
    std::size_t limit = kBest )
{
    ExpectedAssignments result;
    if( cost.n_rows > cost.n_cols ) {
        return result;
    }
    const unsigned states = 1u << cost.n_cols;
    std::vector<std::uint64_t> count( states, 0 );
    std::vector<std::vector<double>> best( states );
    count[0] = 1;
    best[0].push_back( 0.0 );
    const auto better = [sp]( double a, double b ) {
        return sp == SPM::LAP::SP_Min ? a < b : a > b;
    };
    for( unsigned mask = 0; mask < states; ++mask ) {
        const unsigned row = bitCount( mask );
        if( row > cost.n_rows || count[mask] == 0 ) {
            continue;
        }
        auto &sums = best[mask];
        std::sort( sums.begin(), sums.end(), better );
        if( sums.size() > limit ) {
            sums.resize( limit );
        }
        if( row == cost.n_rows ) {
            result.count += count[mask];
            result.costs.insert( result.costs.end(), sums.begin(), sums.end() );
            continue;
        }
        for( unsigned col = 0; col < cost.n_cols; ++col ) {
            if( ( mask & ( 1u << col ) ) != 0 || cost(row, col) == emptyEdge ) {
                continue;
            }
            const unsigned next = mask | ( 1u << col );
            count[next] += count[mask];
            for( double value : sums ) {
                best[next].push_back( value + cost(row, col) );
            }
        }
    }
    std::sort( result.costs.begin(), result.costs.end(), better );
    if( result.costs.size() > limit ) {
        result.costs.resize( limit );
    }
    return result;
}

SPM::Sparse::CMatrixCSR toCsr( const arma::mat &cost )
{
    SPM::Sparse::CMatrixCSR csr;
    SPM::Sparse::MatrixDenseToCSR( cost, csr, emptyEdge, 1e-12 );
    return csr;
}

void checkPrimal( const arma::mat &cost, const SPM::LAP::MurtySolution &actual )
{
    BOOST_REQUIRE_EQUAL( actual.x.n_elem, cost.n_rows );
    std::set<arma::sword> columns;
    double sum = 0.0;
    for( arma::uword row = 0; row < cost.n_rows; ++row ) {
        const arma::sword col = actual.x(row);
        BOOST_REQUIRE_GE( col, 0 );
        BOOST_REQUIRE_LT( static_cast<arma::uword>( col ), cost.n_cols );
        BOOST_CHECK( columns.insert( col ).second );
        BOOST_REQUIRE( cost(row, col) != emptyEdge );
        sum += cost(row, col);
    }
    BOOST_CHECK_SMALL( actual.cost - sum, checkPrecision );
}

void checkDual( const SPM::Sparse::CMatrixCSR &csr, SPM::LAP::TSearchParam sp,
    const SPM::LAP::MurtySolution &actual )
{
    BOOST_REQUIRE_EQUAL( actual.u.n_elem, static_cast<arma::uword>( csr.n_rows() ) );
    BOOST_REQUIRE_EQUAL( actual.v.n_elem, static_cast<arma::uword>( csr.n_cols() ) );
    BOOST_REQUIRE( actual.u.is_finite() );
    BOOST_REQUIRE( actual.v.is_finite() );
    const double sign = sp == SPM::LAP::SP_Min ? 1.0 : -1.0;
    std::set<arma::sword> used;
    for( int row = 0; row < csr.n_rows(); ++row ) {
        used.insert( actual.x(row) );
        for( int t = csr.csr_first[row]; t < csr.csr_first[row + 1]; ++t ) {
            const int col = csr.csr_kk[t];
            const double slack = sign * csr.csr_val[t] - actual.u(row) - actual.v(col);
            BOOST_CHECK_GE( slack, -checkPrecision );
            if( actual.x(row) == col ) {
                BOOST_CHECK_SMALL( slack, checkPrecision );
            }
        }
    }
    if( csr.n_rows() < csr.n_cols() ) {
        for( int col = 0; col < csr.n_cols(); ++col ) {
            BOOST_CHECK_LE( actual.v(col), checkPrecision );
            if( used.count( col ) == 0 ) {
                BOOST_CHECK_SMALL( actual.v(col), checkPrecision );
            }
        }
    }
    BOOST_CHECK_SMALL( sign * actual.cost - arma::accu( actual.u ) - arma::accu( actual.v ), checkPrecision );
}

SPM::LAP::MurtySolution checkJvc( const arma::mat &cost, SPM::LAP::TSearchParam sp,
    const ExpectedAssignments &expected, const SPM::LAP::MurtySolution *parent = nullptr,
    bool reuseAssignment = true )
{
    const auto csr = toCsr( cost );
    SPM::LAP::MurtySolution actual;
    actual.x.set_size( cost.n_rows );
    actual.x.fill( -77 );
    actual.cost = 1234.0;
    if( parent ) {
        actual.u = parent->u;
        actual.v = parent->v;
    }
    const auto before = actual;
    const int rc = parent
        ? SPM::LAP::JVCsparse( csr, sp, 1.0, precision, actual.x, actual.cost, &actual.u, &actual.v,
            reuseAssignment ? &parent->x : nullptr )
        : SPM::LAP::JVCsparse( csr.csr_val, csr.csr_kk, csr.csr_first, sp, 1.0, precision,
            actual.x, actual.cost, &actual.u, &actual.v );
    BOOST_REQUIRE_EQUAL( rc, expected.count == 0 ? 1 : 0 );
    if( rc != 0 ) {
        BOOST_CHECK_EQUAL_COLLECTIONS( actual.x.begin(), actual.x.end(), before.x.begin(), before.x.end() );
        BOOST_CHECK_EQUAL( actual.cost, before.cost );
        BOOST_REQUIRE_EQUAL( actual.u.n_elem, before.u.n_elem );
        BOOST_REQUIRE_EQUAL( actual.v.n_elem, before.v.n_elem );
        if( actual.u.n_elem != 0 ) {
            BOOST_CHECK_EQUAL_COLLECTIONS( actual.u.begin(), actual.u.end(), before.u.begin(), before.u.end() );
        }
        if( actual.v.n_elem != 0 ) {
            BOOST_CHECK_EQUAL_COLLECTIONS( actual.v.begin(), actual.v.end(), before.v.begin(), before.v.end() );
        }
        return actual;
    }
    checkPrimal( cost, actual );
    BOOST_REQUIRE( !expected.costs.empty() );
    BOOST_CHECK_SMALL( actual.cost - expected.costs.front(), checkPrecision );
    checkDual( csr, sp, actual );
    return actual;
}

void checkDense( const arma::mat &cost, SPM::LAP::TSearchParam sp, const ExpectedAssignments &expected,
    double resolution = precision, double infValue = 1.0, arma::uword initialSize = 0 )
{
    BOOST_REQUIRE_GT( expected.count, 0u );
    BOOST_REQUIRE( !expected.costs.empty() );
    SPM::LAP::MurtySolution actual;
    actual.x.set_size( initialSize );
    actual.x.fill( -77 );
    actual.cost = 1234.0;
    SPM::LAP::JVCdense( cost, static_cast<int>( cost.n_rows ), sp, infValue, resolution,
        actual.x, actual.cost );
    checkPrimal( cost, actual );
    double magnitude = 0.0;
    for( double value : cost ) {
        magnitude = std::max( magnitude, std::abs( value ) );
    }
    const double tolerance = std::max( 1e-25, std::max( cost.n_rows * resolution,
        64.0 * std::numeric_limits<double>::epsilon() * cost.n_rows * magnitude ) );
    BOOST_CHECK_SMALL( actual.cost - expected.costs.front(), tolerance );
}

void checkMurty( const arma::mat &cost, SPM::LAP::TSearchParam sp, const ExpectedAssignments &expected )
{
    const auto csr = toCsr( cost );
    SPM::LAP::Murty murty( csr, sp, 1.0, precision );
    std::set<std::vector<arma::sword>> seen;
    SPM::LAP::MurtySolution actual;
    for( double expectedCost : expected.costs ) {
        BOOST_REQUIRE( murty.findNext( actual ) );
        checkPrimal( cost, actual );
        BOOST_CHECK_SMALL( actual.cost - expectedCost, checkPrecision );
        const std::vector<arma::sword> key( actual.x.begin(), actual.x.end() );
        BOOST_CHECK( seen.insert( key ).second );
    }
    if( expected.count <= expected.costs.size() ) {
        actual.x = { -77 };
        actual.cost = 1234.0;
        BOOST_CHECK( !murty.findNext( actual ) );
        BOOST_CHECK( !murty.findNext( actual ) );
        BOOST_CHECK_EQUAL( actual.cost, 1234.0 );
        BOOST_REQUIRE_EQUAL( actual.x.n_elem, 1u );
        BOOST_CHECK_EQUAL( actual.x(0), -77 );
    }
}

void checkMatrix( const arma::mat &cost )
{
    for( auto sp : { SPM::LAP::SP_Min, SPM::LAP::SP_Max } ) {
        BOOST_TEST_CONTEXT( cost.n_rows << 'x' << cost.n_cols << ", sp=" << sp ) {
            const auto expected = expectedAssignments( cost, sp );
            const auto parent = checkJvc( cost, sp, expected );
            checkMurty( cost, sp, expected );
            if( expected.count == 0 ) {
                continue;
            }
            checkJvc( cost, sp, expected, &parent );
            checkJvc( cost, sp, expected, &parent, false );
            arma::mat child = cost;
            const arma::uword split = cost.n_rows / 2;
            child(split, parent.x(split)) = emptyEdge;
            const auto childExpected = expectedAssignments( child, sp, 1 );
            checkJvc( child, sp, childExpected );
            checkJvc( child, sp, childExpected, &parent );
        }
    }
}

arma::mat makeMatrix( unsigned rows, unsigned cols, bool sparse, double scale = 1.0 )
{
    arma::mat cost( rows, cols );
    for( unsigned row = 0; row < rows; ++row ) {
        for( unsigned col = 0; col < cols; ++col ) {
            // Нули, положительные и отрицательные цены; точные четверти в обычном масштабе.
            const int value = static_cast<int>( ( row * 37 + col * 19 + row * col * 11 + rows * 7 + cols ) % 83 ) - 41;
            cost(row, col) = value * 0.25 * scale;
            if( sparse && ( row * 7 + col * 13 + rows + cols ) % 5 < 3
                && row != col && col + 1 != cols ) {
                cost(row, col) = emptyEdge;
            }
        }
    }
    return cost;
}

} // namespace

BOOST_AUTO_TEST_CASE( jvc_murty_previous_4x4_failure_explicit_costs )
{
    const arma::mat cost = {
        { 7, -10, emptyEdge, 2 },
        { 5, -3, -5, -9 },
        { emptyEdge, 9, 3, 10 },
        { emptyEdge, -8, 3, 10 }
    };
    const std::vector<double> sorted = { -7, 2, 4, 8, 8, 10, 17, 17, 19, 21 };
    for( auto sp : { SPM::LAP::SP_Min, SPM::LAP::SP_Max } ) {
        ExpectedAssignments expected;
        expected.count = sorted.size();
        expected.costs = sorted;
        if( sp == SPM::LAP::SP_Max ) {
            std::reverse( expected.costs.begin(), expected.costs.end() );
        }
        const auto oracle = expectedAssignments( cost, sp );
        BOOST_CHECK_EQUAL( oracle.count, expected.count );
        BOOST_CHECK_EQUAL_COLLECTIONS( oracle.costs.begin(), oracle.costs.end(),
            expected.costs.begin(), expected.costs.end() );
        checkJvc( cost, sp, expected );
        checkMurty( cost, sp, expected );
    }
    // Именно эта ветвь останавливала второй findNext(): трём строкам доступны только два столбца.
    const arma::mat infeasibleChild = {
        { emptyEdge, emptyEdge, emptyEdge, 2 },
        { emptyEdge, -3, -5, emptyEdge },
        { emptyEdge, 9, 3, emptyEdge },
        { emptyEdge, -8, 3, emptyEdge }
    };
    arma::mat parentCost = cost;
    parentCost(0, 0) = emptyEdge;
    for( auto sp : { SPM::LAP::SP_Min, SPM::LAP::SP_Max } ) {
        const auto parent = checkJvc( parentCost, sp, expectedAssignments( parentCost, sp ) );
        const ExpectedAssignments none;
        checkJvc( infeasibleChild, sp, none );
        checkJvc( infeasibleChild, sp, none, &parent );
        checkMurty( infeasibleChild, sp, none );
    }
}

BOOST_AUTO_TEST_CASE( jvc_murty_original_paper_10x10_example )
{
    // Пример из исходной статьи Murty (murty1968.pdf). Нули — допустимые стоимости.
    const arma::mat cost = {
        { 7, 51, 52, 87, 38, 60, 74, 66, 0, 20 },
        { 50, 12, 0, 64, 8, 53, 0, 46, 76, 42 },
        { 27, 77, 0, 18, 22, 48, 44, 13, 0, 57 },
        { 62, 0, 3, 8, 5, 6, 14, 0, 26, 39 },
        { 0, 97, 0, 5, 13, 0, 41, 31, 62, 48 },
        { 79, 68, 0, 0, 15, 12, 17, 47, 35, 43 },
        { 76, 99, 48, 27, 34, 0, 0, 0, 28, 0 },
        { 0, 20, 9, 27, 46, 15, 84, 19, 3, 24 },
        { 56, 10, 45, 39, 0, 93, 67, 79, 19, 38 },
        { 27, 0, 39, 53, 46, 24, 69, 46, 23, 1 }
    };
    const arma::ivec optimum = { 8, 6, 2, 7, 5, 3, 9, 0, 4, 1 };
    const auto sp = SPM::LAP::SP_Min;
    const auto expected = expectedAssignments( cost, sp );
    BOOST_REQUIRE_EQUAL( expected.count, 3628800u );
    BOOST_REQUIRE_EQUAL( expected.costs.front(), 0.0 ); // Z_m0
    const auto parent = checkJvc( cost, sp, expected );
    BOOST_CHECK_EQUAL_COLLECTIONS( parent.x.begin(), parent.x.end(), optimum.begin(), optimum.end() );
    checkJvc( cost, sp, expected, &parent );

    const auto csr = toCsr( cost );
    BOOST_CHECK_EQUAL( csr.csr_val.size(), 100u ); // Включая все нулевые рёбра.
    SPM::LAP::Murty murty( csr, sp, 1.0, precision );
    SPM::LAP::MurtySolution first;
    BOOST_REQUIRE( murty.findNext( first ) );
    BOOST_CHECK_EQUAL( first.cost, 0.0 );
    BOOST_CHECK_EQUAL_COLLECTIONS( first.x.begin(), first.x.end(), optimum.begin(), optimum.end() );
    checkMurty( cost, sp, expected ); // Первые 24 решения, включая кратность равных стоимостей.

    // Z_m1..Z_m9 — оптимумы подзадач первого разбиения, а не следующие решения по рангу.
    // В подзадаче split фиксируем предыдущие пары a(1), текущую пару запрещаем.
    // В сообщении указаны 14 и 12 для Z_m2 и Z_m5; для данной матрицы получаются 13 и 15.
    // Назначения ниже (нумерация 1..10) явно подтверждают достижимость каждой стоимости.
    const std::vector<double> partitionCosts = { 10, 13, 14, 1, 15, 53, 45, 47, 56 };
    const std::vector<arma::ivec> partitionAssignments = {
        { 1, 7, 3, 8, 6, 4, 10, 9, 5, 2 },
        { 9, 2, 3, 8, 6, 4, 7, 1, 5, 10 },
        { 9, 7, 8, 2, 3, 4, 6, 1, 5, 10 },
        { 9, 7, 3, 2, 6, 4, 8, 1, 5, 10 },
        { 9, 7, 3, 8, 1, 4, 10, 6, 5, 2 },
        { 9, 7, 3, 8, 6, 5, 4, 1, 2, 10 },
        { 9, 7, 3, 8, 6, 4, 5, 1, 2, 10 },
        { 9, 7, 3, 8, 6, 4, 10, 2, 5, 1 },
        { 9, 7, 3, 8, 6, 4, 10, 1, 2, 5 }
    };
    for( arma::uword split = 0; split < cost.n_rows; ++split ) {
        BOOST_TEST_CONTEXT( "first partition, split=" << split ) {
            arma::mat child = cost;
            for( arma::uword row = 0; row < split; ++row ) {
                child.row( row ).fill( emptyEdge );
                child(row, optimum(row)) = cost(row, optimum(row));
            }
            child(split, optimum(split)) = emptyEdge;
            const auto childExpected = expectedAssignments( child, sp, 1 );
            if( split < partitionCosts.size() ) {
                BOOST_REQUIRE_GT( childExpected.count, 0u );
                BOOST_REQUIRE_EQUAL( childExpected.costs.front(), partitionCosts[split] );
                SPM::LAP::MurtySolution witness;
                witness.x = partitionAssignments[split] - 1;
                witness.cost = partitionCosts[split];
                checkPrimal( child, witness );
            } else {
                // После фиксации девяти пар для последней строки нет альтернативного столбца.
                BOOST_REQUIRE_EQUAL( childExpected.count, 0u );
            }
            checkJvc( child, sp, childExpected );
            checkJvc( child, sp, childExpected, &parent );
        }
    }
}

BOOST_AUTO_TEST_CASE( jvc_murty_dense_all_shapes_1_to_10 )
{
    for( unsigned rows = 1; rows <= 10; ++rows ) {
        for( unsigned cols = 1; cols <= 10; ++cols ) {
            checkMatrix( makeMatrix( rows, cols, false ) );
        }
    }
}

BOOST_AUTO_TEST_CASE( jvc_murty_sparse_all_shapes_1_to_10 )
{
    for( unsigned rows = 1; rows <= 10; ++rows ) {
        for( unsigned cols = 1; cols <= 10; ++cols ) {
            checkMatrix( makeMatrix( rows, cols, true, 1e-6 ) );
        }
    }
}

BOOST_AUTO_TEST_CASE( jvc_murty_vertical_with_explicit_dummy_columns_2_to_10 )
{
    for( unsigned rows = 2; rows <= 10; ++rows ) {
        for( unsigned cols = 1; cols < rows; ++cols ) {
            BOOST_TEST_CONTEXT( "vertical " << rows << 'x' << cols << ", padded to square" ) {
                arma::mat padded( rows, rows, arma::fill::zeros );
                padded.cols( 0, cols - 1 ) = makeMatrix( rows, cols, false );
                checkMatrix( padded );
            }
        }
    }
}

BOOST_AUTO_TEST_CASE( jvc_murty_square_complete_enumeration_up_to_10 )
{
    for( unsigned size = 1; size <= 10; ++size ) {
        arma::mat cost( size, size );
        cost.fill( emptyEdge );
        for( unsigned row = 0; row < size; ++row ) {
            cost(row, row) = static_cast<int>( row ) - 5;
            if( size > 1 ) {
                cost(row, ( row + 1 ) % size) = 7 - static_cast<int>( row );
            }
        }
        const auto expected = expectedAssignments( cost, SPM::LAP::SP_Min );
        BOOST_CHECK_EQUAL( expected.count, size == 1 ? 1u : 2u );
        checkMatrix( cost );
    }
}

BOOST_AUTO_TEST_CASE( jvc_murty_equal_prices_all_shapes_1_to_10 )
{
    for( unsigned rows = 1; rows <= 10; ++rows ) {
        for( unsigned cols = 1; cols <= 10; ++cols ) {
            checkMatrix( arma::mat( rows, cols, arma::fill::zeros ) );
        }
    }
}

BOOST_AUTO_TEST_CASE( jvc_rejects_duplicate_edges_even_in_a_tall_matrix )
{
    const std::vector<double> values = { 1, 2, 3 };
    const std::vector<int> columns = { 0, 0, 0 };
    const std::vector<int> offsets = { 0, 2, 3 };
    arma::ivec x;
    double cost = 0;
    BOOST_CHECK_THROW( SPM::LAP::JVCsparse( values, columns, offsets, SPM::LAP::SP_Min,
        1.0, precision, x, cost ), std::invalid_argument );
}


BOOST_AUTO_TEST_CASE( jvc_murty_horizontal_single_row_1_to_10 )
{
    for( unsigned cols = 1; cols <= 10; ++cols ) {
        arma::mat cost( 1, cols );
        ExpectedAssignments expected;
        expected.count = cols;
        for( unsigned col = 0; col < cols; ++col ) {
            cost(0, col) = static_cast<int>( col ) - 5;
            expected.costs.push_back( cost(0, col) );
        }
        checkJvc( cost, SPM::LAP::SP_Min, expected );
        checkMurty( cost, SPM::LAP::SP_Min, expected );
        std::reverse( expected.costs.begin(), expected.costs.end() );
        checkJvc( cost, SPM::LAP::SP_Max, expected );
        checkMurty( cost, SPM::LAP::SP_Max, expected );
    }
}

BOOST_AUTO_TEST_CASE( jvc_murty_vertical_single_column_1_to_10 )
{
    for( unsigned rows = 1; rows <= 10; ++rows ) {
        arma::mat cost( rows, 1 );
        for( unsigned row = 0; row < rows; ++row ) {
            cost(row, 0) = 11 + row;
        }
        ExpectedAssignments expected;
        if( rows == 1 ) {
            expected.count = 1;
            expected.costs = { 11 };
        }
        for( auto sp : { SPM::LAP::SP_Min, SPM::LAP::SP_Max } ) {
            checkJvc( cost, sp, expected );
            checkMurty( cost, sp, expected );
        }
    }
}

BOOST_AUTO_TEST_CASE( jvc_murty_infeasible_square_sizes_1_to_10 )
{
    for( unsigned size = 1; size <= 10; ++size ) {
        arma::mat cost( size, size );
        cost.fill( emptyEdge );
        if( size > 1 ) {
            cost(0, 0) = -3;
            cost(1, 0) = 5;
        }
        if( size > 2 ) {
            cost(2, 1) = 2;
            cost(2, 2) = -1;
        }
        for( unsigned row = 3; row < size; ++row ) {
            cost(row, row) = row;
        }
        // При size >= 3 нет пустых строк или столбцов, но строки 0 и 1 конкурируют за один столбец.
        for( auto sp : { SPM::LAP::SP_Min, SPM::LAP::SP_Max } ) {
            BOOST_TEST_CONTEXT( "Hall deficiency, size=" << size << ", sp=" << sp ) {
                const auto expected = expectedAssignments( cost, sp );
                BOOST_REQUIRE_EQUAL( expected.count, 0u );
                checkJvc( cost, sp, expected );
                checkMurty( cost, sp, expected );
                const arma::mat feasible = makeMatrix( size, size, false );
                const auto parent = checkJvc( feasible, sp, expectedAssignments( feasible, sp ) );
                checkJvc( cost, sp, expected, &parent );
            }
        }
    }
}

BOOST_AUTO_TEST_CASE( murty_readme_table3_all_22_assignments )
{
    arma::mat cost = {
        { 0.6, 0.3, 0.0, 0.1, 0.0, 0.0 },
        { 0.0, 0.2, 0.7, 0.0, 0.1, 0.0 },
        { 0.3, 0.5, 0.1, 0.0, 0.0, 0.1 }
    };
    // Таблица 3 представляет разреженную задачу: нули не входят в CSR.
    for( double &value : cost ) {
        if( value == 0.0 ) {
            value = emptyEdge;
        }
    }
    ExpectedAssignments expected;
    expected.costs = { 1.8, 1.4, 1.3, 1.3, 1.2, 1.1, 1.1, 0.9, 0.9, 0.9, 0.8,
        0.8, 0.7, 0.7, 0.6, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3 };
    expected.count = expected.costs.size();
    const auto sp = SPM::LAP::SP_Max;
    const auto oracle = expectedAssignments( cost, sp );
    BOOST_REQUIRE_EQUAL( oracle.count, 22u );
    BOOST_REQUIRE_EQUAL( oracle.costs.size(), expected.costs.size() );
    for( std::size_t rank = 0; rank < expected.costs.size(); ++rank ) {
        BOOST_CHECK_SMALL( oracle.costs[rank] - expected.costs[rank], checkPrecision );
    }
    checkMurty( cost, sp, expected );
}

BOOST_AUTO_TEST_CASE( jvc_dense_square_and_horizontal_shapes_1_to_10 )
{
    for( unsigned rows = 1; rows <= 10; ++rows ) {
        for( unsigned cols = rows; cols <= 10; ++cols ) {
            for( double scale : { 1.0, 1e-6, 1e6 } ) {
                const auto cost = makeMatrix( rows, cols, false, scale );
                for( auto sp : { SPM::LAP::SP_Min, SPM::LAP::SP_Max } ) {
                    const auto expected = expectedAssignments( cost, sp, 1 );
                    for( double resolution : { 0.0, precision } ) {
                        BOOST_TEST_CONTEXT( rows << 'x' << cols << ", scale=" << scale
                            << ", sp=" << sp << ", resolution=" << resolution ) {
                            checkDense( cost, sp, expected, resolution );
                        }
                    }
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE( jvc_dense_equal_prices_all_supported_shapes_1_to_10 )
{
    for( unsigned rows = 1; rows <= 10; ++rows ) {
        for( unsigned cols = rows; cols <= 10; ++cols ) {
            for( double value : { -7.0, 0.0, 7.0 } ) {
                arma::mat cost( rows, cols );
                cost.fill( value );
                for( auto sp : { SPM::LAP::SP_Min, SPM::LAP::SP_Max } ) {
                    checkDense( cost, sp, expectedAssignments( cost, sp, 1 ), 0.0 );
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE( jvc_dense_horizontal_uses_all_columns )
{
    // До исправления dim=2 ограничивал поиск левым квадратом и давал минимум 20 вместо 10.
    const arma::mat cost = { { 10, 20, 0 }, { 20, 10, 1 } };
    for( auto sp : { SPM::LAP::SP_Min, SPM::LAP::SP_Max } ) {
        const auto expected = expectedAssignments( cost, sp, 1 );
        BOOST_CHECK_EQUAL( expected.costs.front(), sp == SPM::LAP::SP_Min ? 10.0 : 40.0 );
        checkDense( cost, sp, expected );
    }
}

BOOST_AUTO_TEST_CASE( jvc_dense_preserves_finite_penalties_in_cost )
{
    // Элемент 100 разрешён: оптимум минимума — диагональ стоимостью 100, а не 0 или 120.
    const std::vector<arma::mat> matrices = {
        arma::mat( 1, 1, arma::fill::value( 100.0 ) ),
        arma::mat( 1, 1, arma::fill::value( 100.0 - 5e-10 ) ),
        { { 0, 60 }, { 60, 100 } },
        { { 1, -1e6, -1e5 }, { -1e6, 2, -1e5 }, { -1e5, -1e5, -1e6 } }
    };
    for( const auto &cost : matrices ) {
        for( auto sp : { SPM::LAP::SP_Min, SPM::LAP::SP_Max } ) {
            checkDense( cost, sp, expectedAssignments( cost, sp, 1 ), precision, 100.0 );
        }
    }
}

BOOST_AUTO_TEST_CASE( jvc_dense_auction_large_penalties_terminates )
{
    for( double penalty : { 100.0, 1000.0, 1e12 } ) {
        const arma::mat minimumCost = {
            { 0, 1, penalty, penalty },
            { 0, 2, penalty, penalty },
            { 0, 3, penalty, penalty },
            { penalty, penalty, 0, 1 }
        };
        for( auto sp : { SPM::LAP::SP_Min, SPM::LAP::SP_Max } ) {
            const arma::mat cost = sp == SPM::LAP::SP_Min ? minimumCost : -minimumCost;
            const auto expected = expectedAssignments( cost, sp, 1 );
            BOOST_REQUIRE_EQUAL( expected.costs.front(),
                sp == SPM::LAP::SP_Min ? penalty + 1 : -penalty - 1 );
            for( double resolution : { 0.0, precision } ) {
                checkDense( cost, sp, expected, resolution );
            }
        }
    }
}

BOOST_AUTO_TEST_CASE( jvc_dense_zero_resolution_float_regression )
{
    // Умножение целых на 1e-10 воспроизводит исходные двоичные значения сбойной матрицы.
    arma::mat cost = {
        { 24, -61, 17, -74, -92 },
        { -42, -11, -68, -78, 29 },
        { 12, -64, 22, 55, 13 },
        { -4, -48, 81, -66, 80 },
        { 44, -67, -17, -18, 22 }
    };
    cost *= 1e-10;
    for( auto sp : { SPM::LAP::SP_Min, SPM::LAP::SP_Max } ) {
        checkDense( cost, sp, expectedAssignments( cost, sp, 1 ), 0.0 );
    }
}

BOOST_AUTO_TEST_CASE( jvc_dense_sizes_outputs_and_empty_problems )
{
    const std::vector<arma::mat> matrices = {
        { { 1, 2 }, { 3, 1 } },
        { { 10, 20, 0 }, { 20, 10, 1 } },
        arma::mat( 0, 0 ),
        arma::mat( 0, 3 )
    };
    for( const auto &cost : matrices ) {
        for( auto sp : { SPM::LAP::SP_Min, SPM::LAP::SP_Max } ) {
            const auto expected = expectedAssignments( cost, sp, 1 );
            for( arma::uword initialSize : { 0u, 1u, 2u, 7u } ) {
                checkDense( cost, sp, expected, precision, 1.0, initialSize );
            }
        }
    }
}

BOOST_AUTO_TEST_CASE( jvc_dense_rejects_invalid_input_without_changing_outputs )
{
    const arma::mat valid = { { 1, 2 }, { 3, 4 } };
    const auto reject = []( const arma::mat &cost, int dim, SPM::LAP::TSearchParam sp,
        double infValue, double resolution ) {
        arma::ivec x = { 77, 88 };
        const arma::ivec before = x;
        double value = 1234.0;
        BOOST_CHECK_THROW( SPM::LAP::JVCdense( cost, dim, sp, infValue, resolution, x, value ),
            std::invalid_argument );
        BOOST_CHECK_EQUAL_COLLECTIONS( x.begin(), x.end(), before.begin(), before.end() );
        BOOST_CHECK_EQUAL( value, 1234.0 );
    };
    reject( valid, -1, SPM::LAP::SP_Min, 1.0, precision );
    reject( valid, 0, SPM::LAP::SP_Min, 1.0, precision );
    reject( valid, 1, SPM::LAP::SP_Min, 1.0, precision );
    reject( valid, 3, SPM::LAP::SP_Min, 1.0, precision );
    reject( valid, 2, static_cast<SPM::LAP::TSearchParam>( 2 ), 1.0, precision );
    for( double invalid : { -1.0, 0.0, std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN() } ) {
        reject( valid, 2, SPM::LAP::SP_Min, invalid, precision );
        if( invalid != 0.0 ) {
            reject( valid, 2, SPM::LAP::SP_Min, 1.0, invalid );
        }
    }
    for( double invalid : { std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } ) {
        arma::mat cost = valid;
        cost(0, 0) = invalid;
        reject( cost, 2, SPM::LAP::SP_Min, 1.0, precision );
    }
    for( unsigned rows = 1; rows <= 10; ++rows ) {
        for( unsigned cols = 0; cols < rows; ++cols ) {
            reject( arma::mat( rows, cols, arma::fill::zeros ), rows, SPM::LAP::SP_Min, 1.0, precision );
        }
    }
}

BOOST_AUTO_TEST_CASE( jvc_dense_single_row_and_nearly_equal_prices )
{
    for( unsigned cols = 1; cols <= 10; ++cols ) {
        arma::mat cost( 1, cols );
        for( unsigned col = 0; col < cols; ++col ) {
            cost(0, col) = static_cast<int>( col ) - 5;
        }
        for( auto sp : { SPM::LAP::SP_Min, SPM::LAP::SP_Max } ) {
            checkDense( cost, sp, expectedAssignments( cost, sp, 1 ), 0.0 );
        }
    }
    const arma::mat close = { { 1.5e-9, 0.75e-9, 0.0 } };
    for( auto sp : { SPM::LAP::SP_Min, SPM::LAP::SP_Max } ) {
        checkDense( close, sp, expectedAssignments( close, sp, 1 ), 0.0 );
    }
}
