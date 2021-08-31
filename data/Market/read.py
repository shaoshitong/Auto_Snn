"""
Gallery & Query:    在训练集上进行模型的训练，得到模型后对Query与Gallery中的图片提取特征计算相似度，对于每个Query在Gallery中找出前N个与其相似的图片
Datasets:           单模态ReID数据集->Market-1501、Market-1203、MARS、CUHK03、Duke,跨模态ReID数据集->RegDB、 SYSU-MM01、DBPerson-Recog-DB1
rank_n:             The probability that the top (highest confidence) n graphs in the search results have correct results.[搜索结果中最靠前（置信度最高）的n张图有正
                    确结果的概率。]
Precision & Recall: Precision is how many items (such as documents, web pages, etc.) are retrieved, and recall is how many items are retrieved[Precision就是检
                    索出来的条目（比如：文档、网页等）有多少是准确的，Recall就是所有准确的条目有多少被检索出来了],could explance as A_correct/A_correct && B_correct,B_correc
                    t/A_correct && B_correct
F-score:            2*p*r/(p+r) [recall和precision的调和平均数  2 * P * R / (P + R) ]
mAP:                对于整个gallery，我们的query进行检索，从query的第一个开始，对检索出来的re-person计算召回率和准确率，直到gallery中所检索出来的img都检索完毕，之后以recall为横轴
                    ，precision为纵轴，计算面积，这便是单个Ap,mAp就是取所有query中的均值,其中面积计算公式为=>ap = ap + (recall - old_recall)*((old_precision+precision)/2
                    )
correct-re-distinguish:
                    正确识别：排除同一相机下的同一person，只有不同相机下的同一person算识别成功.
CMC:                对于single gallery shot来说，每一次query，对samples排序，找到匹配上id的gallery后，排除掉同一个camera下同一个id的sample。它的值为rank1,rank2....rankn,分
                    别代表query中所有待识别图像的rankn均值
ROC:                ROC曲线是检测、分类、识别任务中很常用的一项评价指标。曲线上每个点反映着对同一信号刺激的感受性。具体到识别任务中就是，ROC曲线上的每一点反映的是不同的阈值对应的FP（fals
                    e positive）和TP（true positive）之间的关系。TP : True Positive 预测为1，实际也为1；TN：True Nagetive 预测为0，实际也为0,FP：False Positive 预测为1，
                    实际为0的；FN：False Nagetive 预测为0，实际为1的.TPR=TP/(TP+FN)=Recall。FPR=FP/(FP+TN)，FPR即为实际为好人的人中，预测为坏人的人占比。以FPR为x轴，TPR为y轴
                    画图，就得到了ROC曲线。re-id中，采用获得正确分类的概率以及错误分类的概率作为roc的分数
"""
import numpy as np
a=np.ones((30))
a[10:20]=0
b=np.ones((30))
b[15:25]=0
b=b.astype(np.bool)
print(a,b,a[b])
import sparseconvnet.submanifoldConvolution as submanifoldConvolution
submanifoldConvolution
"""
a:1-10 :1 10-20:0 20-30:1
b:1-15 :1 15-25:0 25-30:1
"""
"""
os.path.dirname(os.path.realname(__file__))
whereis cuda
export CUDA_HOME=""
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.1-470.57.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.1-470.57.02-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
"""