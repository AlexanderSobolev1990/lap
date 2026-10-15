//----------------------------------------------------------------------------------------------------------------------
///
/// \file       murty.hpp
/// \brief      Решение k-best задачи о назначениях методом Murty
/// \details    Единичная задача о назначениях решается методом JVC для разреженных матриц
/// \addtogroup spm
/// \{
///

#pragma once

#include <queue>
#include <utility>
#include <vector>

#include <armadillo>

#include "spm/lap/sparse.hpp"
#include "spm/lap/searchparam.hpp"
#include "spm/lap/jvcsparse.hpp"

namespace SPM /// Специальная библиотека программных модулей (СБПМ)
{
namespace LAP /// Решение задачи о назначениях
{
//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Решение метода Murty
///
struct MurtySolution
{
    arma::ivec x; ///< результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке) "rowsol[i] = j" означает, что в i-ой строке выбран j-ый элемент
    arma::vec u; ///< Потенциал строки для минимизации (для SP_Max используется -cost)
    arma::vec v; ///< Потенциал столбца для минимизации (для SP_Max используется -cost)
    double cost = 0.0; ///< Суммарная стоимость назначения
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
    /// \param csr - матрица в CSR формате; должна жить дольше решателя и не изменяться между findNext
    /// \param sp - критерий поиска (минимум/максимум)
    /// \param inf - параметр совместимости с решателем LAP; не ограничивает стоимость назначения
    /// \param res - точность для сравнения двух вещественных чисел
    ///
    Murty( const Sparse::CMatrixCSR & csr, TSearchParam sp, double inf, double res );

    ///
    /// \brief Вызов очередного best-решения задачи
    /// \param out - найденное решение (валидно, если return == true)
    /// \return true - решение найдено, false - решения нет
    ///
    bool findNext( MurtySolution &out );

private:
    struct MurtyNode
    {
        std::vector<int> fixed_col; ///< Фиксированные столбцы
        std::vector<std::pair<int,int>> banned; /// Запрещенные состояния
        MurtySolution sol; /// Решение
        bool feasible = false; ///< Признак существования назначения в узле
        int split_from = 0; /// Разделять начиная с этого индекса
    };

    static bool CompareMax( const MurtyNode &a, const MurtyNode &b );
    static bool CompareMin( const MurtyNode &a, const MurtyNode &b );

    MurtyNode solveNode( const MurtyNode * parent, int split_row ); ///< Решение узла

    const Sparse::CMatrixCSR & csr_; ///< Матрица в CSR формате (Compressed Sparse Row Yale format)
    TSearchParam sp_; ///< Критерий поиска (минимум/максимум)
    double inf_; ///< Параметр совместимости с интерфейсом LAP
    double res_; ///< Точность для сравнения двух вещественных чисел
    bool initialized_; ///< Признак инициализации очереди
    std::priority_queue<
        MurtyNode,
        std::vector<MurtyNode>,
        bool (*)( const MurtyNode &, const MurtyNode & )
    > pq_; ///< Очередь приоритетов
};

} // namespace LAP
} // namespace SPM
/// \}
