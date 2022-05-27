//----------------------------------------------------------------------------------------------------------------------
///
/// \file       lap_jvc_dense.cpp
/// \brief      Решение задачи о назначениях методом JVC для плотных матриц (cтандартная линейная дискретная оптимизационная задача)
/// \date       18.05.22 - создан
/// \author     Соболев А.А.
/// \addtogroup spml
/// \{
///

#include <lap.h>

namespace SPML /// Специальная библиотека программных модулей (СБ ПМ)
{
namespace LAP /// Решение задачи о назначениях
{
//----------------------------------------------------------------------------------------------------------------------
void CAssignmentProblemSolver::JVCsparse( const std::vector<double> &cc, const std::vector<int> &ii,
    const std::vector<int> &kk, int dim, TSearchParam sp, TFindPath fp, double maxcost, double resolution,
    arma::ivec &rowsol, double &lapcost )
{
//    // Если ищем максимум - умножим матрицу на -1
//    std::vector<double> cc_;
//    if( sp == TSearchParam::Max ) { // Поиск минимума/максимума
//        cc_ = cc * (-1.0);
//    } else {
//        cc_ = cc;
//    }

}

} // end namespace LAP
} // end namespace SPML
/// \}

