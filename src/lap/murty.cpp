//----------------------------------------------------------------------------------------------------------------------
///
/// \file       murty.cpp
/// \brief      Решение k-best задачи о назначениях методом Murty
/// \details    Единичная задача о назначениях решается методом JVC для разреженных матриц
/// \date       21.01.26 - создан     
/// \author     Соболев А.А.
/// \addtogroup spml
/// \{
///

#include <murty.hpp>

namespace SPML 
{
namespace LAP 
{

static bool MurtyCmpMax( const MurtyNode &a, const MurtyNode &b ) {
    return a.lb < b.lb; // для поиска Max
}

static bool MurtyCmpMin( const MurtyNode &a, const MurtyNode &b ) {
    return a.lb > b.lb; // для поиска Min
}
  
Murty::Murty( const Sparse::CMatrixCSR& csr, TSearchParam sp, double inf, double res )
    : csr_( csr )
    , sp_( sp )
    , inf_( inf )
    , res_( res )
    , initialized_( false )
    , pq_( ( sp == TSearchParam::SP_Max ) ? MurtyCmpMax : MurtyCmpMin )
{}

MurtyNode Murty::solveNode( const MurtyNode* parent, int split_row )
{
    MurtyNode n;
    n.split_from = split_row + 1;
    const int nr = csr_.n_rows();

    n.fixed_col.assign( nr, -1 );

    if( parent ) {
        // OLD
        // n.fixed_col = parent->fixed_col;

        // NEW
        // Фиксируем все строки < split_row
        for( int i = 0; i < split_row; ++i ) {
            n.fixed_col[i] = parent->sol.x(i);
        }

        // Копируем предыдущие bans
        n.banned = parent->banned;

        // Запрещаем текущее ребро
        int c = parent->sol.x( split_row );
        n.banned.emplace_back( split_row, c );
    }

    LapConstraints lc;
    lc.fixed_col = n.fixed_col.data();
    lc.banned = &n.banned;
    g_lap_constraints = &lc;

    arma::ivec x( nr );
    arma::vec u_out( nr ), v_out( csr_.n_cols() );
    double cost = 0.0;

    arma::vec* u_init = nullptr;
    arma::vec* v_init = nullptr;

    if( parent ) {
        u_init = const_cast<arma::vec*>( &parent->sol.u );
        v_init = const_cast<arma::vec*>( &parent->sol.v );
    }

    // Вызов модифицированного метода JVC c "теплым" стартом
    int rc = Murty_JVCsparse( csr_, sp_, inf_, res_, x, cost, u_init, v_init, &u_out, &v_out );
    g_lap_constraints = nullptr; // Было только для текущей задачи - обнулить!

    if( rc != 0 || cost >= inf_ ) {
        n.sol.cost = inf_;
        return n;
    }

    n.sol.x = x;
    n.sol.u = std::move( u_out );
    n.sol.v = std::move( v_out );
    n.sol.cost = cost;
    n.lb = cost;
    return n;
}

bool Murty::findNext( MurtySolution& out )
{
    if( !initialized_ ) {
        MurtyNode root = solveNode( nullptr, -1 );
        if( root.sol.cost >= inf_ ) {
            return false;
        }
        pq_.push( root );
        initialized_ = true;
    }

    if (pq_.empty()) return false;

    MurtyNode best = pq_.top(); // Лучший - на первой позиции!
    pq_.pop();
    out = best.sol;

    const int nr = best.sol.x.n_elem;
    // for (int i = 0; i < nr; ++i) {
    for( int i = best.split_from; i < nr; ++i ) {
        MurtyNode child = solveNode( &best, i );
        if( child.sol.cost < inf_ ) {
            pq_.push( child );
        }
    }
    return true;
}

} // end namespace LAP
} // end namespace SPML
/// \}
