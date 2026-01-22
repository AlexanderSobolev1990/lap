//----------------------------------------------------------------------------------------------------------------------
///
/// \file       lap_constraints.hpp
/// \brief      Для обеспечения работы метода Murty
/// \date       21.01.25
/// \author     Соболев А.А.
/// \addtogroup spml
/// \{
///

#ifndef SPML_LAP_CONSTRAITS_HPP_
#define SPML_LAP_CONSTRAITS_HPP_

#include <vector>
#include <utility>

namespace SPML /// Специальная библиотека программных модулей (СБ ПМ)
{
namespace LAP /// Решение задачи о назначениях
{
    
struct LapConstraints 
{
    const int* fixed_col = nullptr; // row -> fixed column or -1
    const std::vector<std::pair<int,int>>* banned = nullptr;

    inline bool isAllowed(int i, int j) const 
    {
        if( fixed_col && fixed_col[i] != -1 && fixed_col[i] != j )
            return false;
        if( banned ) {
            for( const auto& p : *banned )
                if( p.first == i && p.second == j )
                    return false;
        }
        return true;
    }
};

extern thread_local const LapConstraints* g_lap_constraints;

inline bool lap_is_allowed( int i, int j )
{
    if( !g_lap_constraints ) {
        return true;
    }
    return g_lap_constraints->isAllowed( i, j );
}

} // namespace LAP
} // namespace SPML
#endif
/// \}
