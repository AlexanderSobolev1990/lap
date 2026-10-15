//----------------------------------------------------------------------------------------------------------------------
///
/// \file       test_spm_lap_murty.cpp
/// \brief      Regression tests for Murty K-best assignment.
///

#define BOOST_TEST_MODULE test_spm_lap_murty

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <stdexcept>
#include <sstream>
#include <vector>

#include <armadillo>
#include <boost/test/included/unit_test.hpp>

#include "spm/lap/murty.hpp"
#include "spm/lap/sparse.hpp"

namespace {

struct BruteSolution
{
    arma::ivec x;
    double cost = 0.0;
};

double denseCost( const arma::mat& cost, const arma::ivec& x )
{
    double result = 0.0;
    for( arma::uword i = 0; i < x.n_elem; ++i ) {
        result += cost( i, x( i ) );
    }
    return result;
}

std::string keyOf( const arma::ivec& x )
{
    std::ostringstream ss;
    for( arma::uword i = 0; i < x.n_elem; ++i ) {
        if( i > 0 ) {
            ss << ',';
        }
        ss << x( i );
    }
    return ss.str();
}

void enumerateAssignments(
    const arma::mat& cost,
    const double empty,
    const int row,
    std::vector<int>& used,
    arma::ivec& current,
    std::vector<BruteSolution>& out )
{
    if( row == static_cast<int>( cost.n_rows ) ) {
        BruteSolution sol;
        sol.x = current;
        sol.cost = denseCost( cost, current );
        out.push_back( sol );
        return;
    }

    for( arma::uword col = 0; col < cost.n_cols; ++col ) {
        if( used[col] != 0 ) {
            continue;
        }
        if( std::abs( cost( row, col ) - empty ) < 1e-12 ) {
            continue;
        }
        used[col] = 1;
        current( row ) = static_cast<int>( col );
        enumerateAssignments( cost, empty, row + 1, used, current, out );
        used[col] = 0;
    }
}

std::vector<BruteSolution> bruteForce(
    const arma::mat& cost,
    const double empty,
    const SPM::LAP::TSearchParam searchParam )
{
    std::vector<BruteSolution> out;
    std::vector<int> used( cost.n_cols, 0 );
    arma::ivec current( cost.n_rows, arma::fill::zeros );
    enumerateAssignments( cost, empty, 0, used, current, out );

    std::sort( out.begin(), out.end(),
        [searchParam]( const BruteSolution& lhs, const BruteSolution& rhs ) {
            // std::sort требует строгого порядка; допуск используется только при проверке результата.
            if( lhs.cost != rhs.cost ) {
                return ( searchParam == SPM::LAP::TSearchParam::SP_Max )
                    ? ( lhs.cost > rhs.cost )
                    : ( lhs.cost < rhs.cost );
            }
            return keyOf( lhs.x ) < keyOf( rhs.x );
        } );
    return out;
}

void checkMurtyAgainstBruteForce(
    const arma::mat& cost,
    const double empty,
    const SPM::LAP::TSearchParam searchParam,
    const int kBest,
    const double infValue = 1e12 )
{
    SPM::Sparse::CMatrixCSR csr;
    SPM::Sparse::MatrixDenseToCSR( cost, csr, empty, 1e-12 );

    const double resolution = 1e-9;
    SPM::LAP::Murty murty( csr, searchParam, infValue, resolution );
    const std::vector<BruteSolution> expected = bruteForce( cost, empty, searchParam );
    std::set<std::string> seen;

    const int count = ( kBest < 0 ) ? static_cast<int>( expected.size() ) : kBest;
    BOOST_REQUIRE_GE( static_cast<int>( expected.size() ), count );
    for( int k = 0; k < count; ++k ) {
        SPM::LAP::MurtySolution actual;
        BOOST_REQUIRE( murty.findNext( actual ) );
        BOOST_CHECK_SMALL( actual.cost - expected[k].cost, 1e-7 );
        BOOST_REQUIRE_EQUAL( actual.x.n_elem, cost.n_rows );
        std::set<int> columns;
        for( arma::uword i = 0; i < actual.x.n_elem; ++i ) {
            BOOST_REQUIRE_GE( actual.x(i), 0 );
            BOOST_REQUIRE_LT( static_cast<arma::uword>( actual.x(i) ), cost.n_cols );
            BOOST_CHECK( std::abs( cost(i, actual.x(i)) - empty ) >= 1e-12 );
            BOOST_CHECK( columns.insert( actual.x(i) ).second );
        }
        BOOST_CHECK_SMALL( actual.cost - denseCost( cost, actual.x ), 1e-7 );
        BOOST_CHECK_EQUAL( seen.insert( keyOf( actual.x ) ).second, true );
    }
    if( kBest < 0 ) {
        SPM::LAP::MurtySolution extra;
        BOOST_CHECK( !murty.findNext( extra ) );
        BOOST_CHECK( !murty.findNext( extra ) );
    }
}

void checkDualCertificate( const SPM::Sparse::CMatrixCSR& csr,
    SPM::LAP::TSearchParam sp, const SPM::LAP::MurtySolution& sol )
{
    const double sign = ( sp == SPM::LAP::TSearchParam::SP_Max ) ? -1.0 : 1.0;
    BOOST_REQUIRE_EQUAL( sol.u.n_elem, static_cast<arma::uword>( csr.n_rows() ) );
    BOOST_REQUIRE_EQUAL( sol.v.n_elem, static_cast<arma::uword>( csr.n_cols() ) );
    std::set<int> used;
    for( int i = 0; i < csr.n_rows(); ++i ) {
        bool matched = false;
        BOOST_CHECK( used.insert( sol.x(i) ).second );
        for( int t = csr.csr_first[i]; t < csr.csr_first[i + 1]; ++t ) {
            const int j = csr.csr_kk[t];
            const double slack = sign * csr.csr_val[t] - sol.u(i) - sol.v(j);
            BOOST_CHECK_GE( slack, -1e-7 );
            if( j == sol.x(i) ) {
                matched = true;
                BOOST_CHECK_SMALL( slack, 1e-7 );
            }
        }
        BOOST_CHECK( matched );
    }
    if( csr.n_rows() < csr.n_cols() ) {
        for( int j = 0; j < csr.n_cols(); ++j ) {
            BOOST_CHECK_LE( sol.v(j), 1e-7 );
            if( used.count( j ) == 0 ) {
                BOOST_CHECK_SMALL( sol.v(j), 1e-7 );
            }
        }
    }
    BOOST_CHECK_SMALL( sign * sol.cost - arma::accu( sol.u ) - arma::accu( sol.v ), 1e-7 );
}

void checkWarmStart( const arma::mat& cost, double empty, SPM::LAP::TSearchParam sp )
{
    SPM::Sparse::CMatrixCSR csr;
    SPM::Sparse::MatrixDenseToCSR( cost, csr, empty, 1e-12 );
    SPM::LAP::MurtySolution parent;
    BOOST_REQUIRE_EQUAL( SPM::LAP::JVCsparse( csr, sp, 10.0, 1e-9, parent.x, parent.cost,
        &parent.u, &parent.v ), 0 );
    checkDualCertificate( csr, sp, parent );
    for( arma::uword i = 0; i < cost.n_rows; ++i ) {
        arma::mat child = cost;
        child(i, parent.x(i)) = empty;
        SPM::Sparse::CMatrixCSR constrained;
        SPM::Sparse::MatrixDenseToCSR( child, constrained, empty, 1e-12 );
        const std::vector<BruteSolution> expected = bruteForce( child, empty, sp );
        SPM::LAP::MurtySolution actual;
        actual.u = parent.u;
        actual.v = parent.v;
        const int rc = SPM::LAP::JVCsparse( constrained, sp, 10.0, 1e-9, actual.x, actual.cost,
            &actual.u, &actual.v, &parent.x );
        BOOST_REQUIRE_EQUAL( rc, expected.empty() ? 1 : 0 );
        if( rc == 0 ) {
            BOOST_CHECK_SMALL( actual.cost - expected.front().cost, 1e-7 );
            checkDualCertificate( constrained, sp, actual );
        }
    }
}

} // namespace

BOOST_AUTO_TEST_CASE( murty_dense_min_matches_bruteforce )
{
    const arma::mat cost = {
        { 9.1, 2.2, 7.3, 8.4 },
        { 6.5, 4.6, 3.7, 7.8 },
        { 5.9, 8.0, 1.1, 8.2 },
        { 7.3, 6.4, 9.5, 4.6 }
    };

    checkMurtyAgainstBruteForce( cost, 0.0, SPM::LAP::TSearchParam::SP_Min, 12 );
}

BOOST_AUTO_TEST_CASE( murty_dense_max_matches_bruteforce )
{
    const arma::mat cost = {
        { 0.61, 0.32, 0.13, 0.24, 0.45 },
        { 0.92, 0.56, 0.79, 0.66, 0.15 },
        { 0.33, 0.88, 0.43, 0.11, 0.58 },
        { 0.77, 0.19, 0.54, 0.82, 0.22 }
    };

    checkMurtyAgainstBruteForce( cost, 0.0, SPM::LAP::TSearchParam::SP_Max, 20 );
}

BOOST_AUTO_TEST_CASE( murty_sparse_min_matches_bruteforce )
{
    const double empty = 1e9;
    arma::mat cost = {
        { 4.1, empty, 7.3, 2.4, empty },
        { empty, 3.6, 5.7, empty, 9.8 },
        { 6.9, 8.0, empty, 1.2, 4.3 },
        { 7.4, empty, 2.5, 6.6, 3.7 }
    };

    checkMurtyAgainstBruteForce( cost, empty, SPM::LAP::TSearchParam::SP_Min, 10 );
}


BOOST_AUTO_TEST_CASE( murty_infeasible_partition_terminates )
{
    const double empty = 1e30;
    const arma::mat cost = {
        { 7, -10, empty, 2 },
        { 5, -3, -5, -9 },
        { empty, 9, 3, 10 },
        { empty, -8, 3, 10 }
    };
    for( auto sp : { SPM::LAP::TSearchParam::SP_Min, SPM::LAP::TSearchParam::SP_Max } ) {
        checkMurtyAgainstBruteForce( cost, empty, sp, -1 );
    }
}

BOOST_AUTO_TEST_CASE( murty_hall_deficiency_without_empty_columns )
{
    const double empty = 1e30;
    const arma::mat cost = {
        { 1, 2, empty, empty, empty },
        { 3, 1, empty, empty, empty },
        { 2, 3, empty, empty, empty },
        { empty, empty, 1, 2, 3 },
        { empty, empty, 3, 1, 2 }
    };
    for( auto sp : { SPM::LAP::TSearchParam::SP_Min, SPM::LAP::TSearchParam::SP_Max } ) {
        checkMurtyAgainstBruteForce( cost, empty, sp, -1 );
    }
}

BOOST_AUTO_TEST_CASE( murty_cost_is_not_a_feasibility_flag )
{
    const arma::mat cost = { { 6, 7 }, { 7, 6 } };
    for( auto sp : { SPM::LAP::TSearchParam::SP_Min, SPM::LAP::TSearchParam::SP_Max } ) {
        checkMurtyAgainstBruteForce( cost, 1e30, sp, -1, 10.0 );
    }
}

BOOST_AUTO_TEST_CASE( murty_warm_start_preserves_optimality_and_duals )
{
    const double empty = 1e30;
    const std::vector<arma::mat> matrices = {
        { { 1, 2, 3 }, { 1, 3, 4 } },
        { { 7, -10, empty, 2 }, { 5, -3, -5, -9 },
          { empty, 9, 3, 10 }, { empty, -8, 3, 10 } },
        // Удаление ребра освобождает столбец с отрицательным потенциалом.
        { { 0, 2, 9 }, { 0, 4, 9 } },
        // После запрета крайнего ребра число столбцов CSR уменьшается.
        { { 1, empty, 8 } }
    };
    for( auto sp : { SPM::LAP::TSearchParam::SP_Min, SPM::LAP::TSearchParam::SP_Max } ) {
        for( const auto& cost : matrices ) {
            checkWarmStart( cost, empty, sp );
        }
    }
}

BOOST_AUTO_TEST_CASE( murty_random_complete_enumeration )
{
    std::mt19937 gen( 761 );
    const double empty = 1e30;
    for( int trial = 0; trial < 100; ++trial ) {
        const int nr = 1 + gen() % 4;
        const int nc = nr + gen() % 3;
        arma::mat cost( nr, nc );
        cost.fill( empty );
        for( int i = 0; i < nr; ++i ) {
            for( int j = 0; j < nc; ++j ) {
                if( gen() % 100 < 75 ) {
                    cost(i, j) = ( static_cast<int>( gen() % 21 ) - 10 ) * ( trial % 2 ? 1e-6 : 1.0 );
                }
            }
        }
        for( auto sp : { SPM::LAP::TSearchParam::SP_Min, SPM::LAP::TSearchParam::SP_Max } ) {
            checkMurtyAgainstBruteForce( cost, empty, sp, -1, 10.0 );
            if( !bruteForce( cost, empty, sp ).empty() ) {
                checkWarmStart( cost, empty, sp );
            }
        }
    }
}


BOOST_AUTO_TEST_CASE( murty_warm_start_keeps_tight_parent_assignments )
{
    SPM::Sparse::CMatrixCSR csr;
    const arma::mat cost = { { 1, 1 }, { 1, 1 } };
    SPM::Sparse::MatrixDenseToCSR( cost, csr, 1e30, 1e-12 );
    const arma::ivec seed = { 1, 0 };
    const arma::vec u = { 5, 5 };
    const arma::vec v = { -4, -4 };
    SPM::LAP::MurtySolution actual;
    actual.u = u;
    actual.v = v;
    BOOST_REQUIRE_EQUAL( SPM::LAP::JVCsparse( csr, SPM::LAP::TSearchParam::SP_Min,
        10.0, 1e-9, actual.x, actual.cost, &actual.u, &actual.v, &seed ), 0 );
    BOOST_CHECK_EQUAL_COLLECTIONS( actual.x.begin(), actual.x.end(), seed.begin(), seed.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( actual.v.begin(), actual.v.end(), v.begin(), v.end() );
    checkDualCertificate( csr, SPM::LAP::TSearchParam::SP_Min, actual );
}

BOOST_AUTO_TEST_CASE( jvc_optional_parameters_and_potentials_only_start )
{
    const arma::mat cost = { { 1, 2, 3 }, { 1, 3, 4 } };
    SPM::Sparse::CMatrixCSR csr;
    SPM::Sparse::MatrixDenseToCSR( cost, csr, 1e30, 1e-12 );
    for( auto sp : { SPM::LAP::SP_Min, SPM::LAP::SP_Max } ) {
        const auto expected = bruteForce( cost, 1e30, sp );
        SPM::LAP::MurtySolution seed;
        // Неподготовленный выходной вектор и обычный вызов без дополнительных аргументов.
        BOOST_REQUIRE_EQUAL( SPM::LAP::JVCsparse( csr, sp, 1.0, 0.0, seed.x, seed.cost ), 0 );
        BOOST_CHECK_SMALL( seed.cost - expected.front().cost, 1e-9 );
        BOOST_REQUIRE_EQUAL( SPM::LAP::JVCsparse( csr.csr_val, csr.csr_kk, csr.csr_first,
            sp, 1.0, 0.0, seed.x, seed.cost, &seed.u, &seed.v ), 0 );
        checkDualCertificate( csr, sp, seed );
        const arma::vec savedU = seed.u, savedV = seed.v;
        // Повторный вызов только с потенциалами, без предыдущего назначения.
        BOOST_REQUIRE_EQUAL( SPM::LAP::JVCsparse( csr, sp, 1.0, 0.0,
            seed.x, seed.cost, &seed.u, &seed.v ), 0 );
        BOOST_CHECK_SMALL( seed.cost - expected.front().cost, 1e-9 );
        checkDualCertificate( csr, sp, seed );
        // Входное назначение может совпадать с выходным: оно читается до записи результата.
        BOOST_REQUIRE_EQUAL( SPM::LAP::JVCsparse( csr, sp, 1.0, 0.0,
            seed.x, seed.cost, &seed.u, &seed.v, &seed.x ), 0 );
        checkDualCertificate( csr, sp, seed );
        arma::vec onlyU, onlyV;
        BOOST_REQUIRE_EQUAL( SPM::LAP::JVCsparse( csr, sp, 1.0, 0.0,
            seed.x, seed.cost, &onlyU ), 0 );
        BOOST_REQUIRE_EQUAL( SPM::LAP::JVCsparse( csr, sp, 1.0, 0.0,
            seed.x, seed.cost, nullptr, &onlyV ), 0 );
        BOOST_CHECK_EQUAL_COLLECTIONS( onlyU.begin(), onlyU.end(), savedU.begin(), savedU.end() );
        BOOST_CHECK_EQUAL_COLLECTIONS( onlyV.begin(), onlyV.end(), savedV.begin(), savedV.end() );
        seed.u.set_size( 1 );
        BOOST_REQUIRE_EQUAL( SPM::LAP::JVCsparse( csr, sp, 1.0, 0.0,
            seed.x, seed.cost, &seed.u, &seed.v ), 0 );
        checkDualCertificate( csr, sp, seed );
    }
}

BOOST_AUTO_TEST_CASE( jvc_rejects_malformed_input_and_preserves_outputs_on_infeasibility )
{
    SPM::Sparse::CMatrixCSR csr;
    arma::ivec x = { 7, 8 };
    arma::vec u = { 3, 4 }, v = { 5, 6 };
    double cost = 99;
    BOOST_CHECK_THROW( SPM::LAP::JVCsparse( csr, SPM::LAP::SP_Min, 1, 0, x, cost ), std::length_error );
    csr.csr_first = { 0, 0, 0 };
    BOOST_CHECK_EQUAL( SPM::LAP::JVCsparse( csr, SPM::LAP::SP_Min, 1, 0, x, cost, &u, &v ), 1 );
    BOOST_CHECK_EQUAL( x(0), 7 );
    BOOST_CHECK_EQUAL( cost, 99 );
    BOOST_CHECK_EQUAL( u(0), 3 );
    BOOST_CHECK_EQUAL( v(0), 5 );
    csr.csr_val = { 1, 2 };
    csr.csr_kk = { 0, 1 };
    csr.csr_first = { 0, 3, 2 };
    BOOST_CHECK_THROW( SPM::LAP::JVCsparse( csr, SPM::LAP::SP_Min, 1, 0, x, cost ), std::invalid_argument );
    csr.csr_first = { 0, 2 };
    csr.csr_kk = { 0, 0 };
    BOOST_CHECK_THROW( SPM::LAP::JVCsparse( csr, SPM::LAP::SP_Min, 1, 0, x, cost ), std::invalid_argument );
    csr.csr_first = { 0, 1, 2 };
    csr.csr_kk = { 0, 1 };
    BOOST_CHECK_THROW( SPM::LAP::JVCsparse( csr, SPM::LAP::SP_Min, -1, 0, x, cost ), std::invalid_argument );
    BOOST_CHECK_THROW( SPM::LAP::JVCsparse( csr, SPM::LAP::SP_Min, 1, -1, x, cost ), std::invalid_argument );
    BOOST_CHECK_THROW( SPM::LAP::JVCsparse( csr, SPM::LAP::SP_Min, 1, 0, x, cost, &u, &u ), std::invalid_argument );
}


BOOST_AUTO_TEST_CASE( jvc_murty_exact_order_for_nearly_equal_costs )
{
    const arma::mat cost = { { 1.5e-9, 0.75e-9, 0.0 } };
    SPM::Sparse::CMatrixCSR csr;
    SPM::Sparse::MatrixDenseToCSR( cost, csr, 1e30, 1e-12 );
    for( auto sp : { SPM::LAP::SP_Min, SPM::LAP::SP_Max } ) {
        const auto expected = bruteForce( cost, 1e30, sp );
        BOOST_REQUIRE_EQUAL( expected.size(), 3u );
        const std::vector<double> costs = sp == SPM::LAP::SP_Min
            ? std::vector<double>{ 0.0, 0.75e-9, 1.5e-9 }
            : std::vector<double>{ 1.5e-9, 0.75e-9, 0.0 };
        arma::ivec x;
        double value = 0;
        BOOST_REQUIRE_EQUAL( SPM::LAP::JVCsparse( csr, sp, 1.0, 0.0, x, value ), 0 );
        BOOST_CHECK_SMALL( value - costs.front(), 1e-20 );
        SPM::LAP::Murty murty( csr, sp, 1.0, 0.0 );
        SPM::LAP::MurtySolution actual;
        for( std::size_t k = 0; k < costs.size(); ++k ) {
            BOOST_CHECK_SMALL( expected[k].cost - costs[k], 1e-20 );
            BOOST_REQUIRE( murty.findNext( actual ) );
            BOOST_CHECK_SMALL( actual.cost - costs[k], 1e-20 );
        }
        BOOST_CHECK( !murty.findNext( actual ) );
    }
}
