# Python数据预处理和可视化

## Jax
### Jax特点
- Jax.numpy与numpy不能互转
- Jax会有各种key，保证程序可复现性

  ```
  from jax import random
  key = random.PRNGKey(0)
  x = random.normal(key, (10, ))
  ```

- Jax可以根据环境调整
- Jax JIT第一次运行时编译
- JIT修饰符，返回新函数
- traceback
- static_argnums
    - 第二维操作
    - for循环自动按raw loop
    - 预留shape大小空间
    - 不能使用参数key
- 没有真正的稀疏矩阵
- 图网络
    - mask矩阵表示领域关系
    - 稀疏矩阵
- numpy中的特殊函数take(to do)?
- .block_until_ready()真正触发运行
- pmap 多个XLA设备并行
- jax scipy科学计算，numpy包，image图像包
- lax包中有的不用numpy
- 神经网络只能有cond, scan可以替换fori_loop
- while_loop设计无法得到循环次数
- tree_multimap不同层设计不同学习率
- dot_general
   - contracting dimensions [m, n] [n, p]-> n消失的维度
   - batch dimension [b, m, n], [b, n, p]-> b相同的维度，对应位置乘
      
   ```
   from jax.lax import dot_general as dot
   x = random.normal(key, (64, 100, 10))
   w = random.normal(key, (10, 15))
   dot(x, w, (((2,), (0,)),((), ())).shape
   ```
- tree_map 多层map相同操作
- tree_multimap 多层map不同key不同操作

   ```
   @jax.
   ```

- cond 输出必须是相同大小
- fori_loop和while_loop 不能backward，要用scan
- associaive_scan 累加

## Pandas
### Pandas特点
- pandas engine='python'(可以为c)
- 多个矩阵concat，统一处理
- columns.tolist()查看所有列
- 数据处理的坑
    - 读入错误的数据，检查方法，对比行列，行列错误。
    - sql数据库，导入csv，会串行列。
    - 数据导出很大，解编码错误
    - 训练集和测试集串行
    - 文本中\t \n 预处理清除
    - concat 行对应错误
- pandas只能在内存中
- pandas默认逗号，文本\t
- 列 key, unique() 查看值的情况
- value_counts 查看值分布
- -set删除部分列
- select筛选
    - loc选多个条件

        ```
        data['VehicleAge'][data['']>3].count()
        data.loc[data['VechileAge']>3, 'VechicleAge'].count()
        ```

    - iloc选择index
- map对列处理
- groupby+agg聚合操作，as_index=False否则多层级index， 可以reset_index不建议reset，需要index配对
- pandas join, left join, outer join，建议left join，否则会丢失原数据
- pd.join不建议，建议pd.merge
- 原表与聚合之后列表，join会重名，自动重命名

## R and dplyr
### R and dplyr特点
- colnames查看数据的列
- %>% 管道
- dplyr.tidyverse.org/reference/index.html 参考文档
- R所有数据读入内存
- python和r最好不要互相传数据，最好写入硬盘，使用csv读入
- docker限制每个人的内存

## matplotlib

### matplotlib特点
- line chart／scatter chart最多看三维
- line chart / scatter chart会漏掉趋势，但是图能帮助发现异常值
- 去掉异常值再进行分析
- boxplot，可以看到平均方差，查看数据分布
- smoothing可以看到趋势
- log可以不停看到结果，实时监控查看会不会崩溃。 


## 闲聊
- 上海地区建行反洗钱表一个4PB
- 建行集群600个512G内存128core MPP!!!
- select * from 反洗钱表，全行奔溃 (spark sql -> hdfs ?)，建行的sql引擎似乎太垃圾了，居然不会自动OOM挂掉
- 生产环境与测试环境数据分开，自己装好docker，限制内存，跑摊玩玩，跑测试库，最后部署生产环境
- 检查数据质量
- 研究异常值的来源
- 博士工资最低，实际是只记载了在校博士
- 大量转入转出，是在洗钱
- 大厂很多人一月份断供

