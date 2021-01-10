#python性能调优

##基本准则

###查找性能瓶颈

###优化优先级
1. 算法本身
    a. 算法在当前问题上的性能，例如，收敛速度
    b. 算法的时间复杂度
2. 优化实现
    a. 使用更接近底层的语言
    b. 进行并行化处理，python中ray，c中openmp 


## Cython

### Cython的使用场景
- cython可以看做简洁的c来使用
- cython做各种预处理，将指针传给c程序，在c程序中进行复杂的计算过程

### Cython使用
- cython -a -a输出注释，用户可以看到cython对对应代码行生成的c代码
- typed memory view 类似传入指针，如下，可传入numpy数组

   ```
   def shannon_entropy_mv(double[::1] p_x):
   ```
   
- vector<vector<int>> 类似动态存储的二维数据块，数据不一定是连续存储的
- 二维数组存储方式，行优先或列优先，列优先 fortran，行优先 c，根据操作选在按行或按列存储

## OpenMP

### OpenMP使用

- cython中的prange等价于parallel_for
  
  ```
  from cython.parallel import prange
  ```
- openmp不能调用python中的东西，比如list
- cython中传递的数组，没有shape
- cython的检查关闭
- nogil标记

  ```
  cdef void parallel_v2(double[:,:] x, const long nrow, const long column) nogil:
  ```
  
- range也不能用，range也是python的类


### Cython注意点
- cython从c程序中返回结果，在cython创建资源，传递给c
- cython程序和c程序，在哪里创建临时资源，在哪里销毁资源，c中需要注意内存管理

## VTune的使用
- 需要付费，或者，注册学生帐号，暂时无法使用
- 运行过程中，需要修改各种flag，修改过后可能会显示无法保存，但实际已保存
- 主要观看指标，vectorization，utilization
- 可以查看函数调用，function/call stack，并展示源代码，以及每行的运行时间，以及对应的汇编语句



