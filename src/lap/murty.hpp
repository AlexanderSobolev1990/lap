//----------------------------------------------------------------------------------------------------------------------
///
/// \file       murty.hpp
/// \brief      Решение k-best задачи о назначениях методом Murty
/// \details    Единичная задача о назначениях решается методом JVC для разреженных матриц
/// \date       21.01.26 - создан
/// \author     Соболев А.А.
/// \addtogroup spml
/// \{
///

#ifndef SPML_MURTY_HPP_
#define SPML_MURTY_HPP_

#include <armadillo>
#include <queue>
#include <vector>

#include <lap_constraints.hpp>
#include <sparse.hpp>
#include <searchparam.hpp>
#include <murty_jvc_sparse.hpp>

namespace SPML /// Специальная библиотека программных модулей (СБПМ) 
{
namespace LAP /// Решение задачи о назначениях 
{
//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Решение метода Murty
///
struct MurtySolution {
    arma::ivec x; ///< результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке) "rowsol[i] = j" означает, что в i-ой строке выбран j-ый элемент
    arma::vec u; ///< Двойственная переменная строки
    arma::vec v; ///< Двойственная переменная столбца
    double cost; ///< Суммарная стоимость назначения
};

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Узел решения
///
struct MurtyNode {
    std::vector<int> fixed_col; ///< Фиксированные столбцы
    std::vector<std::pair<int,int>> banned; /// Запрещенные состояния
    MurtySolution sol; /// Решение
    double lb; ///< Нижняя граница
    int split_from = 0; /// Разделять начиная с этого индекса
};

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Класс решения K-best задачи методом Murty
///
class Murty 
{
public:
    ///
    /// \brief Конструктор метода Murty
    /// \param csr - матрица в CSR формате (Compressed Sparse Row Yale format)
    /// \param sp - критерий поиска (минимум/максимум)
    /// \param inf - большое положительное число
    /// \param res - точность для сравнения двух вещественных чисел
    ///
    Murty( const Sparse::CMatrixCSR& csr, TSearchParam sp, double inf, double res );
    
    ///
    /// \brief Вызов очередного best-решения задачи     
    /// \param out - найденное решение (валидно, если return == true)
    /// \return true - решение найдено, false - решения нет
    ///
    bool findNext( MurtySolution &out );

private:    
    MurtyNode solveNode( const MurtyNode* parent, int split_row ); ///< Решение узла

    const Sparse::CMatrixCSR& csr_; ///< Матрица в CSR формате (Compressed Sparse Row Yale format)
    TSearchParam sp_; ///< Критерий поиска (минимум/максимум)
    double inf_; ///< Большое положительное число
    double res_; ///< Точность для сравнения двух вещественных чисел 
    bool initialized_; ///< Признак "тёплого" старта
    std::priority_queue<
        MurtyNode,
        std::vector<MurtyNode>,
        bool(*)( const MurtyNode&, const MurtyNode& )
    > pq_; ///< Очередь приоритетов
};

} // namespace LAP
} // namespace SPML
#endif
/// \}