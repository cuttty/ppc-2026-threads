s #pragma once
#include <string>
#include <vector>

#include "task/include/task.hpp"

    namespace frolova_s_radix_sort_double {

  using InType = std::vector<double>;
  using OutType = std::vector<double>;
  using TestType = std::string;
  using BaseTask = ppc::task::Task<InType, OutType>;

  class RadixSortDouble : public BaseTask {
   public:
    static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
      return ppc::task::TypeOfTask::kALL;
    }
    explicit RadixSortDouble(const InType &in);

   private:
    bool ValidationImpl() override;
    bool PreProcessingImpl() override;
    bool RunImpl() override;
    bool PostProcessingImpl() override;
  };

}  // namespace frolova_s_radix_sort_double
