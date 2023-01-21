# Memory Mechanism in Video Detection

## 1. Introduction

### 1.1. Memory

Recent state-of-the-art Video Object Segmentation (VOS) methods use attention to link representations of past frames stored in the ``feature memory`` with features extracted from the newly observed query frame which needs to be segmented. Despite the high performance of these methods, they require a ``large amount of GPU memory`` to store past frame representations. In practice, they usually struggle to handle videos longer than a minute on consumer-grade hardware.

### 1.2. Related Work


## 2. Paper List

<!DOCTYPE html>
<html><head>
		<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
		<title>Zotero 报告</title>
		<link rel="stylesheet" type="text/css" href="data:text/css;base64,Ym9keSB7CgliYWNrZ3JvdW5kOiB3aGl0ZTsKfQoKYSB7Cgl0ZXh0LWRlY29yYXRpb246IHVuZGVybGluZTsKfQoKYm9keSB7CglwYWRkaW5nOiAwOwp9Cgp1bC5yZXBvcnQgbGkuaXRlbSB7Cglib3JkZXItdG9wOiA0cHggc29saWQgIzU1NTsKCXBhZGRpbmctdG9wOiAxZW07CglwYWRkaW5nLWxlZnQ6IDFlbTsKCXBhZGRpbmctcmlnaHQ6IDFlbTsKCW1hcmdpbi1ib3R0b206IDJlbTsKfQoKaDEsIGgyLCBoMywgaDQsIGg1LCBoNiB7Cglmb250LXdlaWdodDogbm9ybWFsOwp9CgpoMiB7CgltYXJnaW46IDAgMCAuNWVtOwp9CgpoMi5wYXJlbnRJdGVtIHsKCWZvbnQtd2VpZ2h0OiBib2xkOwoJZm9udC1zaXplOiAxZW07CglwYWRkaW5nOiAwIDAgLjVlbTsKCWJvcmRlci1ib3R0b206IDFweCBzb2xpZCAjY2NjOwp9CgovKiBJZiBjb21iaW5pbmcgY2hpbGRyZW4sIGRpc3BsYXkgcGFyZW50IHNsaWdodGx5IGxhcmdlciAqLwp1bC5yZXBvcnQuY29tYmluZUNoaWxkSXRlbXMgaDIucGFyZW50SXRlbSB7Cglmb250LXNpemU6IDEuMWVtOwoJcGFkZGluZy1ib3R0b206IC43NWVtOwoJbWFyZ2luLWJvdHRvbTogLjRlbTsKfQoKaDIucGFyZW50SXRlbSAudGl0bGUgewoJZm9udC13ZWlnaHQ6IG5vcm1hbDsKfQoKaDMgewoJbWFyZ2luLWJvdHRvbTogLjZlbTsKCWZvbnQtd2VpZ2h0OiBib2xkICFpbXBvcnRhbnQ7Cglmb250LXNpemU6IDFlbTsKCWRpc3BsYXk6IGJsb2NrOwp9CgovKiBNZXRhZGF0YSB0YWJsZSAqLwp0aCB7Cgl2ZXJ0aWNhbC1hbGlnbjogdG9wOwoJdGV4dC1hbGlnbjogcmlnaHQ7Cgl3aWR0aDogMTUlOwoJd2hpdGUtc3BhY2U6IG5vd3JhcDsKfQoKdGQgewoJcGFkZGluZy1sZWZ0OiAuNWVtOwp9CgoKdWwucmVwb3J0LCB1bC5ub3RlcywgdWwudGFncyB7CglsaXN0LXN0eWxlOiBub25lOwoJbWFyZ2luLWxlZnQ6IDA7CglwYWRkaW5nLWxlZnQ6IDA7Cn0KCi8qIFRhZ3MgKi8KaDMudGFncyB7Cglmb250LXNpemU6IDEuMWVtOwp9Cgp1bC50YWdzIHsKCWxpbmUtaGVpZ2h0OiAxLjc1ZW07CglsaXN0LXN0eWxlOiBub25lOwp9Cgp1bC50YWdzIGxpIHsKCWRpc3BsYXk6IGlubGluZTsKfQoKdWwudGFncyBsaTpub3QoOmxhc3QtY2hpbGQpOmFmdGVyIHsKCWNvbnRlbnQ6ICcsICc7Cn0KCgovKiBDaGlsZCBub3RlcyAqLwpoMy5ub3RlcyB7Cglmb250LXNpemU6IDEuMWVtOwp9Cgp1bC5ub3RlcyB7CgltYXJnaW4tYm90dG9tOiAxLjJlbTsKfQoKdWwubm90ZXMgPiBsaTpmaXJzdC1jaGlsZCBwIHsKCW1hcmdpbi10b3A6IDA7Cn0KCnVsLm5vdGVzID4gbGkgewoJcGFkZGluZzogLjdlbSAwOwp9Cgp1bC5ub3RlcyA+IGxpOm5vdCg6bGFzdC1jaGlsZCkgewoJYm9yZGVyLWJvdHRvbTogMXB4ICNjY2Mgc29saWQ7Cn0KCgp1bC5ub3RlcyA+IGxpIHA6Zmlyc3QtY2hpbGQgewoJbWFyZ2luLXRvcDogMDsKfQoKdWwubm90ZXMgPiBsaSBwOmxhc3QtY2hpbGQgewoJbWFyZ2luLWJvdHRvbTogMDsKfQoKLyogQWRkIHF1b3RhdGlvbiBtYXJrcyBhcm91bmQgYmxvY2txdW90ZSAqLwp1bC5ub3RlcyA+IGxpIGJsb2NrcXVvdGUgcDpub3QoOmVtcHR5KTpiZWZvcmUsCmxpLm5vdGUgYmxvY2txdW90ZSBwOm5vdCg6ZW1wdHkpOmJlZm9yZSB7Cgljb250ZW50OiAn4oCcJzsKfQoKdWwubm90ZXMgPiBsaSBibG9ja3F1b3RlIHA6bm90KDplbXB0eSk6bGFzdC1jaGlsZDphZnRlciwKbGkubm90ZSBibG9ja3F1b3RlIHA6bm90KDplbXB0eSk6bGFzdC1jaGlsZDphZnRlciB7Cgljb250ZW50OiAn4oCdJzsKfQoKLyogUHJlc2VydmUgd2hpdGVzcGFjZSBvbiBwbGFpbnRleHQgbm90ZXMgKi8KdWwubm90ZXMgbGkgcC5wbGFpbnRleHQsIGxpLm5vdGUgcC5wbGFpbnRleHQsIGRpdi5ub3RlIHAucGxhaW50ZXh0IHsKCXdoaXRlLXNwYWNlOiBwcmUtd3JhcDsKfQoKLyogRGlzcGxheSB0YWdzIHdpdGhpbiBjaGlsZCBub3RlcyBpbmxpbmUgKi8KdWwubm90ZXMgaDMudGFncyB7CglkaXNwbGF5OiBpbmxpbmU7Cglmb250LXNpemU6IDFlbTsKfQoKdWwubm90ZXMgaDMudGFnczphZnRlciB7Cgljb250ZW50OiAnICc7Cn0KCnVsLm5vdGVzIHVsLnRhZ3MgewoJZGlzcGxheTogaW5saW5lOwp9Cgp1bC5ub3RlcyB1bC50YWdzIGxpOm5vdCg6bGFzdC1jaGlsZCk6YWZ0ZXIgewoJY29udGVudDogJywgJzsKfQoKCi8qIENoaWxkIGF0dGFjaG1lbnRzICovCmgzLmF0dGFjaG1lbnRzIHsKCWZvbnQtc2l6ZTogMS4xZW07Cn0KCnVsLmF0dGFjaG1lbnRzIGxpIHsKCXBhZGRpbmctdG9wOiAuNWVtOwp9Cgp1bC5hdHRhY2htZW50cyBkaXYubm90ZSB7CgltYXJnaW4tbGVmdDogMmVtOwp9Cgp1bC5hdHRhY2htZW50cyBkaXYubm90ZSBwOmZpcnN0LWNoaWxkIHsKCW1hcmdpbi10b3A6IC43NWVtOwp9CgpkaXYgdGFibGUgewoJYm9yZGVyLWNvbGxhcHNlOiBjb2xsYXBzZTsKfQoKZGl2IHRhYmxlIHRkLCBkaXYgdGFibGUgdGggewoJYm9yZGVyOiAxcHggI2NjYyBzb2xpZDsKCWJvcmRlci1jb2xsYXBzZTogY29sbGFwc2U7Cgl3b3JkLWJyZWFrOiBicmVhay1hbGw7Cn0KCmRpdiB0YWJsZSB0ZCBwOmVtcHR5OjphZnRlciwgZGl2IHRhYmxlIHRoIHA6ZW1wdHk6OmFmdGVyIHsKCWNvbnRlbnQ6ICJcMDBhMCI7Cn0KCmRpdiB0YWJsZSB0ZCAqOmZpcnN0LWNoaWxkLCBkaXYgdGFibGUgdGggKjpmaXJzdC1jaGlsZCB7CgltYXJnaW4tdG9wOiAwOwp9CgpkaXYgdGFibGUgdGQgKjpsYXN0LWNoaWxkLCBkaXYgdGFibGUgdGggKjpsYXN0LWNoaWxkIHsKCW1hcmdpbi1ib3R0b206IDA7Cn0K">
		<link rel="stylesheet" type="text/css" media="screen,projection" href="data:text/css;base64,LyogR2VuZXJpYyBzdHlsZXMgKi8KYm9keSB7Cglmb250OiA2Mi41JSBHZW9yZ2lhLCBUaW1lcywgc2VyaWY7Cgl3aWR0aDogNzgwcHg7CgltYXJnaW46IDAgYXV0bzsKfQoKaDIgewoJZm9udC1zaXplOiAxLjVlbTsKCWxpbmUtaGVpZ2h0OiAxLjVlbTsKCWZvbnQtZmFtaWx5OiBHZW9yZ2lhLCBUaW1lcywgc2VyaWY7Cn0KCnAgewoJbGluZS1oZWlnaHQ6IDEuNWVtOwp9CgphOmxpbmssIGE6dmlzaXRlZCB7Cgljb2xvcjogIzkwMDsKfQoKYTpob3ZlciwgYTphY3RpdmUgewoJY29sb3I6ICM3Nzc7Cn0KCgp1bC5yZXBvcnQgewoJZm9udC1zaXplOiAxLjRlbTsKCXdpZHRoOiA2ODBweDsKCW1hcmdpbjogMCBhdXRvOwoJcGFkZGluZzogMjBweCAyMHB4Owp9CgovKiBNZXRhZGF0YSB0YWJsZSAqLwp0YWJsZSB7Cglib3JkZXI6IDFweCAjY2NjIHNvbGlkOwoJb3ZlcmZsb3c6IGF1dG87Cgl3aWR0aDogMTAwJTsKCW1hcmdpbjogLjFlbSBhdXRvIC43NWVtOwoJcGFkZGluZzogMC41ZW07Cn0K">
		<link rel="stylesheet" type="text/css" media="print" href="data:text/css;base64,Ym9keSB7Cglmb250OiAxMnB0ICJUaW1lcyBOZXcgUm9tYW4iLCBUaW1lcywgR2VvcmdpYSwgc2VyaWY7CgltYXJnaW46IDA7Cgl3aWR0aDogYXV0bzsKCWNvbG9yOiBibGFjazsKfQoKLyogUGFnZSBCcmVha3MgKHBhZ2UtYnJlYWstaW5zaWRlIG9ubHkgcmVjb2duaXplZCBieSBPcGVyYSkgKi8KaDEsIGgyLCBoMywgaDQsIGg1LCBoNiB7CglwYWdlLWJyZWFrLWFmdGVyOiBhdm9pZDsKCXBhZ2UtYnJlYWstaW5zaWRlOiBhdm9pZDsKfQoKdWwsIG9sLCBkbCB7CglwYWdlLWJyZWFrLWluc2lkZTogYXZvaWQ7Cgljb2xvci1hZGp1c3Q6IGV4YWN0Owp9CgpoMiB7Cglmb250LXNpemU6IDEuM2VtOwoJbGluZS1oZWlnaHQ6IDEuM2VtOwp9CgphIHsKCWNvbG9yOiAjMDAwOwoJdGV4dC1kZWNvcmF0aW9uOiBub25lOwp9Cg==">
	</head>
	<body>
		<ul class="report combineChildItems">
			<li id="item_WY3GCAYB" class="item preprint">
			<h2>XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model</h2>
				<table>
					<tbody><tr>
						<th>类型</th>
						<td>预印本</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Ho Kei Cheng</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Alexander G. Schwing</td>
					</tr>
					<tr>
					<th>摘要</th>
						<td>We devise XMem. Inspired by the Atkinson–Shiﬀrin memory model 
[1], we introduce memory stores with diﬀerent temporal scales and equip 
them with a memory reading operation for high-quality video object 
segmentation on both long and short videos.</td>
					</tr>
					<tr>
					<th>日期</th>
						<td>2022-07-18</td>
					</tr>
					<tr>
					<th>语言</th>
						<td>en</td>
					</tr>
					<tr>
					<th>短标题</th>
						<td>XMem</td>
					</tr>
					<tr>
					<th>馆藏目录</th>
						<td>arXiv.org</td>
					</tr>
					<tr>
					<th>URL</th>
						<td><a href="http://arxiv.org/abs/2207.07115">http://arxiv.org/abs/2207.07115</a></td>
					</tr>
					<tr>
					<th>访问时间</th>
						<td>2023/1/16 下午1:06:26</td>
					</tr>
					<tr>
					<th>其它</th>
						<td>arXiv:2207.07115 [cs]</td>
					</tr>
					<tr>
					<th>仓库</th>
						<td>arXiv</td>
					</tr>
					<tr>
					<th>存档ID</th>
						<td>arXiv:2207.07115</td>
					</tr>
					<tr>
					<th>添加日期</th>
						<td>2023/1/16 下午1:06:26</td>
					</tr>
					<tr>
					<th>修改日期</th>
						<td>2023/1/16 下午1:06:26</td>
					</tr>
				</tbody></table>
				<h3 class="tags">标签：</h3>
				<ul class="tags">
					<li>Computer Science - Computer Vision and Pattern Recognition</li>
				</ul>
				<h3 class="notes">笔记：</h3>
				<ul class="notes">
					<li id="item_YFLTVIQ6">
<p class="plaintext">Comment: Accepted to ECCV 2022. Project page: https://hkchengrex.github.io/XMem</p>
					</li>
				</ul>
				<h3 class="attachments">附件</h3>
				<ul class="attachments">
					<li id="item_LJYAU2EF">Cheng 和 Schwing - 2022 - XMem Long-Term Video Object Segmentation with an .pdf					</li>
				</ul>
			</li>


			<li id="item_SUP3GAR9" class="item conferencePaper">
			<h2>Video Object Segmentation with Dynamic Memory Networks and Adaptive Object Alignment</h2>
				<table>
					<tbody><tr>
						<th>类型</th>
						<td>会议论文</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Shuxian Liang</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Xu Shen</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Jianqiang Huang</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Xian-Sheng Hua</td>
					</tr>
					<tr>
					<th>摘要</th>
						<td>In this paper, we propose a novel solution for objectmatching 
based semi-supervised video object segmentation, where the target object
 masks in the ﬁrst frame are provided. Existing object-matching based 
methods focus on the matching between the raw object features of the 
current frame and the ﬁrst/previous frames. However, two issues are 
still not solved by these object-matching based methods. As the 
appearance of the video object changes drastically over time, 1) unseen 
parts/details of the object present in the current frame, resulting in 
incomplete annotation in the ﬁrst annotated frame (e.g. view/scale 
changes). 2) even for the seen parts/details of the object in the 
current frame, their positions change relatively (e.g. pose 
changes/camera motion), leading to a misalignment for the object 
matching. To obtain the complete information of the target object, we 
propose a novel object-based dynamic memory network that exploits visual
 contents of all the past frames. To solve the misalignment problem 
caused by position changes of visual contents, we propose an adaptive 
object alignment module by incorporating a region translation function 
that aligns object proposals towards templates in the feature space. Our
 method achieves state-of-the-art results on latest benchmark datasets 
DAVIS 2017 (J of 81.4% and F of 87.5% on the validation set) and 
YouTube-VOS (the overall score of 82.7% on the validation set) with a 
very efﬁcient inference time (0.16 second/frame on DAVIS 2017 validation
 set). Code is available at: https://github.com/ liang4sx/DMN-AOA.</td>
					</tr>
					<tr>
					<th>日期</th>
						<td>10/2021</td>
					</tr>
					<tr>
					<th>语言</th>
						<td>en</td>
					</tr>
					<tr>
					<th>馆藏目录</th>
						<td>DOI.org (Crossref)</td>
					</tr>
					<tr>
					<th>URL</th>
						<td><a href="https://ieeexplore.ieee.org/document/9710441/">https://ieeexplore.ieee.org/document/9710441/</a></td>
					</tr>
					<tr>
					<th>访问时间</th>
						<td>2023/1/21 上午11:31:05</td>
					</tr>
					<tr>
					<th>地点</th>
						<td>Montreal, QC, Canada</td>
					</tr>
					<tr>
					<th>出版社</th>
						<td>IEEE</td>
					</tr>
					<tr>
					<th>ISBN</th>
						<td>978-1-66542-812-5</td>
					</tr>
					<tr>
					<th>页码</th>
						<td>8045-8054</td>
					</tr>
					<tr>
					<th>会议论文集标题</th>
						<td>2021 IEEE/CVF International Conference on Computer Vision (ICCV)</td>
					</tr>
					<tr>
					<th>会议名称</th>
						<td>2021 IEEE/CVF International Conference on Computer Vision (ICCV)</td>
					</tr>
					<tr>
					<th>DOI</th>
						<td><a href="http://doi.org/10.1109/ICCV48922.2021.00796">10.1109/ICCV48922.2021.00796</a></td>
					</tr>
					<tr>
					<th>添加日期</th>
						<td>2023/1/21 上午11:31:06</td>
					</tr>
					<tr>
					<th>修改日期</th>
						<td>2023/1/21 上午11:31:06</td>
					</tr>
				</tbody></table>
				<h3 class="attachments">附件</h3>
				<ul class="attachments">
					<li id="item_5BUXZPYL">Liang 等 - 2021 - Video Object Segmentation with Dynamic Memory Netw.pdf					</li>
				</ul>
			</li>


			<li id="item_Y6WI8SRM" class="item journalArticle">
			<h2>Video Object Segmentation with Adaptive Feature Bank and Uncertain-Region Reﬁnement</h2>
				<table>
					<tbody><tr>
						<th>类型</th>
						<td>期刊文章</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Yongqing Liang</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Xin Li</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Navid Jafari</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Qin Chen</td>
					</tr>
					<tr>
					<th>摘要</th>
						<td>This paper presents a new matching-based framework for 
semi-supervised video object segmentation (VOS). Recently, 
state-of-the-art VOS performance has been achieved by matching-based 
algorithms, in which feature banks are created to store features for 
region matching and classiﬁcation. However, how to effectively organize 
information in the continuously growing feature bank remains 
underexplored, and this leads to an inefﬁcient design of the bank. We 
introduced an adaptive feature bank update scheme to dynamically absorb 
new features and discard obsolete features. We also designed a new 
conﬁdence loss and a ﬁnegrained segmentation module to enhance the 
segmentation accuracy in uncertain regions. On public benchmarks, our 
algorithm outperforms existing state-of-thearts.</td>
					</tr>
					<tr>
					<th>语言</th>
						<td>en</td>
					</tr>
					<tr>
					<th>馆藏目录</th>
						<td>Zotero</td>
					</tr>
					<tr>
					<th>添加日期</th>
						<td>2023/1/21 上午11:31:10</td>
					</tr>
					<tr>
					<th>修改日期</th>
						<td>2023/1/21 上午11:31:10</td>
					</tr>
				</tbody></table>
				<h3 class="attachments">附件</h3>
				<ul class="attachments">
					<li id="item_BMP8ADQT">Liang 等 - Video Object Segmentation with Adaptive Feature Ba.pdf					</li>
				</ul>
			</li>


			<li id="item_GSEGH4C7" class="item conferencePaper">
			<h2>Video Object Segmentation Using Space-Time Memory Networks</h2>
				<table>
					<tbody><tr>
						<th>类型</th>
						<td>会议论文</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Seoung Wug Oh</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Joon-Young Lee</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Ning Xu</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Seon Joo Kim</td>
					</tr>
					<tr>
					<th>摘要</th>
						<td>We propose a novel solution for semi-supervised video object 
segmentation. By the nature of the problem, available cues (e.g. video 
frame(s) with object masks) become richer with the intermediate 
predictions. However, the existing methods are unable to fully exploit 
this rich source of information. We resolve the issue by leveraging 
memory networks and learn to read relevant information from all 
available sources. In our framework, the past frames with object masks 
form an external memory, and the current frame as the query is segmented
 using the mask information in the memory. Speciﬁcally, the query and 
the memory are densely matched in the feature space, covering all the 
space-time pixel locations in a feed-forward fashion. Contrast to the 
previous approaches, the abundant use of the guidance information allows
 us to better handle the challenges such as appearance changes and 
occlussions. We validate our method on the latest benchmark sets and 
achieved the state-of-the-art performance (overall score of 79.4 on 
Youtube-VOS val set, J of 88.7 and 79.2 on DAVIS 2016/2017 val set 
respectively) while having a fast runtime (0.16 second/frame on DAVIS 
2016 val set).</td>
					</tr>
					<tr>
					<th>日期</th>
						<td>10/2019</td>
					</tr>
					<tr>
					<th>语言</th>
						<td>en</td>
					</tr>
					<tr>
					<th>馆藏目录</th>
						<td>DOI.org (Crossref)</td>
					</tr>
					<tr>
					<th>URL</th>
						<td><a href="https://ieeexplore.ieee.org/document/9008790/">https://ieeexplore.ieee.org/document/9008790/</a></td>
					</tr>
					<tr>
					<th>访问时间</th>
						<td>2023/1/21 上午11:31:15</td>
					</tr>
					<tr>
					<th>地点</th>
						<td>Seoul, Korea (South)</td>
					</tr>
					<tr>
					<th>出版社</th>
						<td>IEEE</td>
					</tr>
					<tr>
					<th>ISBN</th>
						<td>978-1-72814-803-8</td>
					</tr>
					<tr>
					<th>页码</th>
						<td>9225-9234</td>
					</tr>
					<tr>
					<th>会议论文集标题</th>
						<td>2019 IEEE/CVF International Conference on Computer Vision (ICCV)</td>
					</tr>
					<tr>
					<th>会议名称</th>
						<td>2019 IEEE/CVF International Conference on Computer Vision (ICCV)</td>
					</tr>
					<tr>
					<th>DOI</th>
						<td><a href="http://doi.org/10.1109/ICCV.2019.00932">10.1109/ICCV.2019.00932</a></td>
					</tr>
					<tr>
					<th>添加日期</th>
						<td>2023/1/21 上午11:31:15</td>
					</tr>
					<tr>
					<th>修改日期</th>
						<td>2023/1/21 上午11:31:16</td>
					</tr>
				</tbody></table>
				<h3 class="attachments">附件</h3>
				<ul class="attachments">
					<li id="item_4FTGAM2F">Oh 等 - 2019 - Video Object Segmentation Using Space-Time Memory .pdf					</li>
				</ul>
			</li>


			<li id="item_7HP923XW" class="item conferencePaper">
			<h2>Video Object Segmentation Using Global and Instance Embedding Learning</h2>
				<table>
					<tbody><tr>
						<th>类型</th>
						<td>会议论文</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Wenbin Ge</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Xiankai Lu</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Jianbing Shen</td>
					</tr>
					<tr>
					<th>日期</th>
						<td>6/2021</td>
					</tr>
					<tr>
					<th>语言</th>
						<td>en</td>
					</tr>
					<tr>
					<th>馆藏目录</th>
						<td>DOI.org (Crossref)</td>
					</tr>
					<tr>
					<th>URL</th>
						<td><a href="https://ieeexplore.ieee.org/document/9577683/">https://ieeexplore.ieee.org/document/9577683/</a></td>
					</tr>
					<tr>
					<th>访问时间</th>
						<td>2023/1/21 上午11:31:00</td>
					</tr>
					<tr>
					<th>地点</th>
						<td>Nashville, TN, USA</td>
					</tr>
					<tr>
					<th>出版社</th>
						<td>IEEE</td>
					</tr>
					<tr>
					<th>ISBN</th>
						<td>978-1-66544-509-2</td>
					</tr>
					<tr>
					<th>页码</th>
						<td>16831-16840</td>
					</tr>
					<tr>
					<th>会议论文集标题</th>
						<td>2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</td>
					</tr>
					<tr>
					<th>会议名称</th>
						<td>2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</td>
					</tr>
					<tr>
					<th>DOI</th>
						<td><a href="http://doi.org/10.1109/CVPR46437.2021.01656">10.1109/CVPR46437.2021.01656</a></td>
					</tr>
					<tr>
					<th>添加日期</th>
						<td>2023/1/21 上午11:31:00</td>
					</tr>
					<tr>
					<th>修改日期</th>
						<td>2023/1/21 上午11:31:00</td>
					</tr>
				</tbody></table>
				<h3 class="attachments">附件</h3>
				<ul class="attachments">
					<li id="item_HV4MUEYF">Ge 等 - 2021 - Video Object Segmentation Using Global and Instanc.pdf					</li>
				</ul>
			</li>


			<li id="item_QTSD9PF4" class="item conferencePaper">
			<h2>SSTVOS: Sparse Spatiotemporal Transformers for Video Object Segmentation</h2>
				<table>
					<tbody><tr>
						<th>类型</th>
						<td>会议论文</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Brendan Duke</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Abdalla Ahmed</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Christian Wolf</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Parham Aarabi</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Graham W. Taylor</td>
					</tr>
					<tr>
					<th>摘要</th>
						<td>In this paper we introduce a Transformer-based approach to 
video object segmentation (VOS). To address compounding error and 
scalability issues of prior work, we propose a scalable, end-to-end 
method for VOS called Sparse Spatiotemporal Transformers (SST). SST 
extracts per-pixel representations for each object in a video using 
sparse attention over spatiotemporal features. Our attention-based 
formulation for VOS allows a model to learn to attend over a history of 
multiple frames and provides suitable inductive bias for performing 
correspondence-like computations necessary for solving motion 
segmentation. We demonstrate the effectiveness of attention-based over 
recurrent networks in the spatiotemporal domain. Our method achieves 
competitive results on YouTube-VOS and DAVIS 2017 with improved 
scalability and robustness to occlusions compared with the state of the 
art. Code is available at https: //github.com/dukebw/SSTVOS.</td>
					</tr>
					<tr>
					<th>日期</th>
						<td>6/2021</td>
					</tr>
					<tr>
					<th>语言</th>
						<td>en</td>
					</tr>
					<tr>
					<th>短标题</th>
						<td>SSTVOS</td>
					</tr>
					<tr>
					<th>馆藏目录</th>
						<td>DOI.org (Crossref)</td>
					</tr>
					<tr>
					<th>URL</th>
						<td><a href="https://ieeexplore.ieee.org/document/9578213/">https://ieeexplore.ieee.org/document/9578213/</a></td>
					</tr>
					<tr>
					<th>访问时间</th>
						<td>2023/1/21 上午11:30:57</td>
					</tr>
					<tr>
					<th>地点</th>
						<td>Nashville, TN, USA</td>
					</tr>
					<tr>
					<th>出版社</th>
						<td>IEEE</td>
					</tr>
					<tr>
					<th>ISBN</th>
						<td>978-1-66544-509-2</td>
					</tr>
					<tr>
					<th>页码</th>
						<td>5908-5917</td>
					</tr>
					<tr>
					<th>会议论文集标题</th>
						<td>2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</td>
					</tr>
					<tr>
					<th>会议名称</th>
						<td>2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</td>
					</tr>
					<tr>
					<th>DOI</th>
						<td><a href="http://doi.org/10.1109/CVPR46437.2021.00585">10.1109/CVPR46437.2021.00585</a></td>
					</tr>
					<tr>
					<th>添加日期</th>
						<td>2023/1/21 上午11:30:57</td>
					</tr>
					<tr>
					<th>修改日期</th>
						<td>2023/1/21 上午11:30:57</td>
					</tr>
				</tbody></table>
				<h3 class="attachments">附件</h3>
				<ul class="attachments">
					<li id="item_XQ6VQKDV">Duke 等 - 2021 - SSTVOS Sparse Spatiotemporal Transformers for Vid.pdf					</li>
				</ul>
			</li>


			<li id="item_BAB5BPUE" class="item journalArticle">
			<h2>Rethinking Space-Time Networks with Improved Memory Coverage for Efﬁcient Video Object Segmentation</h2>
				<table>
					<tbody><tr>
						<th>类型</th>
						<td>期刊文章</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Ho Kei Cheng</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Yu-Wing Tai</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Chi-Keung Tang</td>
					</tr>
					<tr>
					<th>摘要</th>
						<td>This paper presents a simple yet effective approach to 
modeling space-time correspondences in the context of video object 
segmentation. Unlike most existing approaches, we establish 
correspondences directly between frames without reencoding the mask 
features for every object, leading to a highly efﬁcient and robust 
framework. With the correspondences, every node in the current query 
frame is inferred by aggregating features from the past in an 
associative fashion. We cast the aggregation process as a voting problem
 and ﬁnd that the existing inner-product afﬁnity leads to poor use of 
memory with a small (ﬁxed) subset of memory nodes dominating the votes, 
regardless of the query. In light of this phenomenon, we propose using 
the negative squared Euclidean distance instead to compute the 
afﬁnities. We validate that every memory node now has a chance to 
contribute, and experimentally show that such diversiﬁed voting is 
beneﬁcial to both memory efﬁciency and inference accuracy. The synergy 
of correspondence networks and diversiﬁed voting works exceedingly well,
 achieves new state-of-the-art results on both DAVIS and YouTubeVOS 
datasets while running signiﬁcantly faster at 20+ FPS for multiple 
objects without bells and whistles.</td>
					</tr>
					<tr>
					<th>语言</th>
						<td>en</td>
					</tr>
					<tr>
					<th>馆藏目录</th>
						<td>Zotero</td>
					</tr>
					<tr>
					<th>添加日期</th>
						<td>2023/1/21 上午11:31:13</td>
					</tr>
					<tr>
					<th>修改日期</th>
						<td>2023/1/21 上午11:31:13</td>
					</tr>
				</tbody></table>
				<h3 class="attachments">附件</h3>
				<ul class="attachments">
					<li id="item_YRJCEN2U">Cheng 等 - Rethinking Space-Time Networks with Improved Memor.pdf					</li>
				</ul>
			</li>


			<li id="item_H5J2EFTW" class="item journalArticle">
			<h2>Reliable Propagation-Correction Modulation for Video Object Segmentation</h2>
				<table>
					<tbody><tr>
						<th>类型</th>
						<td>期刊文章</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Xiaohao Xu</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Jinglu Wang</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Xiao Li</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Yan Lu</td>
					</tr>
					<tr>
					<th>摘要</th>
						<td>Error propagation is a general but crucial problem in online 
semi-supervised video object segmentation. We aim to suppress error 
propagation through a correction mechanism with high reliability. The 
key insight is to disentangle the correction from the conventional mask 
propagation process with reliable cues. We introduce two modulators, 
propagation and correction modulators, to separately perform 
channel-wise re-calibration on the target frame embeddings according to 
local temporal correlations and reliable references respectively. 
Specifically, we assemble the modulators with a cascaded 
propagation-correction scheme. This avoids overriding the effects of the
 reliable correction modulator by the propagation modulator. Although 
the reference frame with the ground truth label provides reliable cues, 
it could be very different from the target frame and introduce uncertain
 or incomplete correlations. We augment the reference cues by 
supplementing reliable feature patches to a maintained pool, thus 
offering more comprehensive and expressive object representations to the
 modulators. In addition, a reliability filter is designed to retrieve 
reliable patches and pass them in subsequent frames. Our model achieves 
state-of-the-art performance on YouTube-VOS18/19 and DAVIS17-Val/Test 
benchmarks. Extensive experiments demonstrate that the correction 
mechanism provides considerable performance gain by fully utilizing 
reliable guidance.</td>
					</tr>
					<tr>
					<th>日期</th>
						<td>2022-06-28</td>
					</tr>
					<tr>
					<th>语言</th>
						<td>en</td>
					</tr>
					<tr>
					<th>馆藏目录</th>
						<td>DOI.org (Crossref)</td>
					</tr>
					<tr>
					<th>URL</th>
						<td><a href="https://ojs.aaai.org/index.php/AAAI/article/view/20200">https://ojs.aaai.org/index.php/AAAI/article/view/20200</a></td>
					</tr>
					<tr>
					<th>访问时间</th>
						<td>2023/1/21 上午11:30:52</td>
					</tr>
					<tr>
					<th>卷次</th>
						<td>36</td>
					</tr>
					<tr>
					<th>页码</th>
						<td>2946-2954</td>
					</tr>
					<tr>
					<th>期刊</th>
						<td>Proceedings of the AAAI Conference on Artificial Intelligence</td>
					</tr>
					<tr>
					<th>DOI</th>
						<td><a href="http://doi.org/10.1609/aaai.v36i3.20200">10.1609/aaai.v36i3.20200</a></td>
					</tr>
					<tr>
					<th>期号</th>
						<td>3</td>
					</tr>
					<tr>
					<th>刊名缩写</th>
						<td>AAAI</td>
					</tr>
					<tr>
					<th>ISSN</th>
						<td>2374-3468, 2159-5399</td>
					</tr>
					<tr>
					<th>添加日期</th>
						<td>2023/1/21 上午11:30:52</td>
					</tr>
					<tr>
					<th>修改日期</th>
						<td>2023/1/21 上午11:30:52</td>
					</tr>
				</tbody></table>
				<h3 class="attachments">附件</h3>
				<ul class="attachments">
					<li id="item_797S5RDF">Xu 等 - 2022 - Reliable Propagation-Correction Modulation for Vid.pdf					</li>
				</ul>
			</li>


			<li id="item_FWT2BMWY" class="item conferencePaper">
			<h2>Joint Inductive and Transductive Learning for Video Object Segmentation</h2>
				<table>
					<tbody><tr>
						<th>类型</th>
						<td>会议论文</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Yunyao Mao</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Ning Wang</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Wengang Zhou</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Houqiang Li</td>
					</tr>
					<tr>
					<th>摘要</th>
						<td>Semi-supervised video object segmentation is a task of 
segmenting the target object in a video sequence given only a mask 
annotation in the first frame. The limited information available makes 
it an extremely challenging task. Most previous best-performing methods 
adopt matchingbased transductive reasoning or online inductive learning.
 Nevertheless, they are either less discriminative for similar instances
 or insufficient in the utilization of spatio-temporal information. In 
this work, we propose to integrate transductive and inductive learning 
into a unified framework to exploit the complementarity between them for
 accurate and robust video object segmentation. The proposed approach 
consists of two functional branches. The transduction branch adopts a 
lightweight transformer architecture to aggregate rich spatio-temporal 
cues while the induction branch performs online inductive learning to 
obtain discriminative target information. To bridge these two diverse 
branches, a two-head label encoder is introduced to learn the suitable 
target prior for each of them. The generated mask encodings are further 
forced to be disentangled to better retain their complementarity. 
Extensive experiments on several prevalent benchmarks show that, without
 the need of synthetic training data, the proposed approach sets a 
series of new state-of-the-art records. Code is available at 
https://github.com/maoyunyao/JOINT.</td>
					</tr>
					<tr>
					<th>日期</th>
						<td>10/2021</td>
					</tr>
					<tr>
					<th>语言</th>
						<td>en</td>
					</tr>
					<tr>
					<th>馆藏目录</th>
						<td>DOI.org (Crossref)</td>
					</tr>
					<tr>
					<th>URL</th>
						<td><a href="https://ieeexplore.ieee.org/document/9711303/">https://ieeexplore.ieee.org/document/9711303/</a></td>
					</tr>
					<tr>
					<th>访问时间</th>
						<td>2023/1/21 上午11:31:08</td>
					</tr>
					<tr>
					<th>地点</th>
						<td>Montreal, QC, Canada</td>
					</tr>
					<tr>
					<th>出版社</th>
						<td>IEEE</td>
					</tr>
					<tr>
					<th>ISBN</th>
						<td>978-1-66542-812-5</td>
					</tr>
					<tr>
					<th>页码</th>
						<td>9650-9659</td>
					</tr>
					<tr>
					<th>会议论文集标题</th>
						<td>2021 IEEE/CVF International Conference on Computer Vision (ICCV)</td>
					</tr>
					<tr>
					<th>会议名称</th>
						<td>2021 IEEE/CVF International Conference on Computer Vision (ICCV)</td>
					</tr>
					<tr>
					<th>DOI</th>
						<td><a href="http://doi.org/10.1109/ICCV48922.2021.00953">10.1109/ICCV48922.2021.00953</a></td>
					</tr>
					<tr>
					<th>添加日期</th>
						<td>2023/1/21 上午11:31:08</td>
					</tr>
					<tr>
					<th>修改日期</th>
						<td>2023/1/21 上午11:31:08</td>
					</tr>
				</tbody></table>
				<h3 class="attachments">附件</h3>
				<ul class="attachments">
					<li id="item_WXIMJV27">Mao 等 - 2021 - Joint Inductive and Transductive Learning for Vide.pdf					</li>
				</ul>
			</li>


			<li id="item_VZNWF3AY" class="item conferencePaper">
			<h2>Fast Video Object Segmentation With Temporal Aggregation Network and Dynamic Template Matching</h2>
				<table>
					<tbody><tr>
						<th>类型</th>
						<td>会议论文</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Xuhua Huang</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Jiarui Xu</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Yu-Wing Tai</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Chi-Keung Tang</td>
					</tr>
					<tr>
					<th>摘要</th>
						<td>Signiﬁcant progress has been made in Video Object Segmentation
 (VOS), the video object tracking task in its ﬁnest level. While the VOS
 task can be naturally decoupled into image semantic segmentation and 
video object tracking, signiﬁcantly much more research effort has been 
made in segmentation than tracking. In this paper, we introduce 
“tracking-by-detection” into VOS which can coherently integrates 
segmentation into tracking, by proposing a new temporal aggregation 
network and a novel dynamic time-evolving template matching mechanism to
 achieve signiﬁcantly improved performance. Notably, our method is 
entirely online and thus suitable for oneshot learning, and our 
end-to-end trainable model allows multiple object segmentation in one 
forward pass. We achieve new state-of-the-art performance on the DAVIS 
benchmark without complicated bells and whistles in both speed and 
accuracy, with a speed of 0.14 second per frame and J &amp;F measure of 
75.9% respectively. Project page is available at 
https://xuhuaking.github. io/Fast-VOS-DTTM-TAN/.</td>
					</tr>
					<tr>
					<th>日期</th>
						<td>6/2020</td>
					</tr>
					<tr>
					<th>语言</th>
						<td>en</td>
					</tr>
					<tr>
					<th>馆藏目录</th>
						<td>DOI.org (Crossref)</td>
					</tr>
					<tr>
					<th>URL</th>
						<td><a href="https://ieeexplore.ieee.org/document/9156779/">https://ieeexplore.ieee.org/document/9156779/</a></td>
					</tr>
					<tr>
					<th>访问时间</th>
						<td>2023/1/21 上午11:31:03</td>
					</tr>
					<tr>
					<th>地点</th>
						<td>Seattle, WA, USA</td>
					</tr>
					<tr>
					<th>出版社</th>
						<td>IEEE</td>
					</tr>
					<tr>
					<th>ISBN</th>
						<td>978-1-72817-168-5</td>
					</tr>
					<tr>
					<th>页码</th>
						<td>8876-8886</td>
					</tr>
					<tr>
					<th>会议论文集标题</th>
						<td>2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</td>
					</tr>
					<tr>
					<th>会议名称</th>
						<td>2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</td>
					</tr>
					<tr>
					<th>DOI</th>
						<td><a href="http://doi.org/10.1109/CVPR42600.2020.00890">10.1109/CVPR42600.2020.00890</a></td>
					</tr>
					<tr>
					<th>添加日期</th>
						<td>2023/1/21 上午11:31:03</td>
					</tr>
					<tr>
					<th>修改日期</th>
						<td>2023/1/21 上午11:31:03</td>
					</tr>
				</tbody></table>
				<h3 class="attachments">附件</h3>
				<ul class="attachments">
					<li id="item_VQTCE82P">Huang 等 - 2020 - Fast Video Object Segmentation With Temporal Aggre.pdf					</li>
				</ul>
			</li>


			<li id="item_9MHCJRJS" class="item preprint">
			<h2>Fast Video Object Segmentation using the Global Context Module</h2>
				<table>
					<tbody><tr>
						<th>类型</th>
						<td>预印本</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Yu Li</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Zhuoran Shen</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Ying Shan</td>
					</tr>
					<tr>
					<th>摘要</th>
						<td>We developed a real-time, high-quality semi-supervised video 
object segmentation algorithm. Its accuracy is on par with the most 
accurate, time-consuming online-learning model, while its speed is 
similar to the fastest template-matching method with sub-optimal 
accuracy. The core component of the model is a novel global context 
module that eﬀectively summarizes and propagates information through the
 entire video. Compared to previous approaches that only use one frame 
or a few frames to guide the segmentation of the current frame, the 
global context module uses all past frames. Unlike the previous 
state-of-the-art space-time memory network that caches a memory at each 
spatio-temporal position, the global context module uses a ﬁxed-size 
feature representation. Therefore, it uses constant memory regardless of
 the video length and costs substantially less memory and computation. 
With the novel module, our model achieves top performance on standard 
benchmarks at a real-time speed.</td>
					</tr>
					<tr>
					<th>日期</th>
						<td>2020-07-17</td>
					</tr>
					<tr>
					<th>语言</th>
						<td>en</td>
					</tr>
					<tr>
					<th>馆藏目录</th>
						<td>arXiv.org</td>
					</tr>
					<tr>
					<th>URL</th>
						<td><a href="http://arxiv.org/abs/2001.11243">http://arxiv.org/abs/2001.11243</a></td>
					</tr>
					<tr>
					<th>访问时间</th>
						<td>2023/1/21 上午11:31:22</td>
					</tr>
					<tr>
					<th>其它</th>
						<td>arXiv:2001.11243 [cs]</td>
					</tr>
					<tr>
					<th>仓库</th>
						<td>arXiv</td>
					</tr>
					<tr>
					<th>存档ID</th>
						<td>arXiv:2001.11243</td>
					</tr>
					<tr>
					<th>添加日期</th>
						<td>2023/1/21 上午11:31:22</td>
					</tr>
					<tr>
					<th>修改日期</th>
						<td>2023/1/21 上午11:31:22</td>
					</tr>
				</tbody></table>
				<h3 class="tags">标签：</h3>
				<ul class="tags">
					<li>Computer Science - Computer Vision and Pattern Recognition</li>
					<li>I.4.6</li>
				</ul>
				<h3 class="notes">笔记：</h3>
				<ul class="notes">
					<li id="item_872C7TWA">
<p class="plaintext">Comment: To appear at ECCV 2020</p>
					</li>
				</ul>
				<h3 class="attachments">附件</h3>
				<ul class="attachments">
					<li id="item_NUSWR9WQ">Li 等 - 2020 - Fast Video Object Segmentation using the Global Co.pdf					</li>
				</ul>
			</li>


			<li id="item_GT39T3P3" class="item conferencePaper">
			<h2>CapsuleVOS: Semi-Supervised Video Object Segmentation Using Capsule Routing</h2>
				<table>
					<tbody><tr>
						<th>类型</th>
						<td>会议论文</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Kevin Duarte</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Yogesh Rawat</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Mubarak Shah</td>
					</tr>
					<tr>
					<th>摘要</th>
						<td>In this work we propose a capsule-based approach for 
semi-supervised video object segmentation. Current video object 
segmentation methods are frame-based and often require optical ﬂow to 
capture temporal consistency across frames which can be difﬁcult to 
compute. To this end, we propose a video based capsule network, 
CapsuleVOS, which can segment several frames at once conditioned on a 
reference frame and segmentation mask. This conditioning is performed 
through a novel routing algorithm for attention-based efﬁcient capsule 
selection. We address two challenging issues in video object 
segmentation: 1) segmentation of small objects and 2) occlusion of 
objects across time. The issue of segmenting small objects is addressed 
with a zooming module which allows the network to process small spatial 
regions of the video. Apart from this, the framework utilizes a novel 
memory module based on recurrent networks which helps in tracking 
objects when they move out of frame or are occluded. The network is 
trained end-to-end and we demonstrate its effectiveness on two benchmark
 video object segmentation datasets; it outperforms current ofﬂine 
approaches on the Youtube-VOS dataset while having a run-time that is 
almost twice as fast as competing methods. The code is publicly 
available at https://github.com/KevinDuarte/CapsuleVOS.</td>
					</tr>
					<tr>
					<th>日期</th>
						<td>10/2019</td>
					</tr>
					<tr>
					<th>语言</th>
						<td>en</td>
					</tr>
					<tr>
					<th>短标题</th>
						<td>CapsuleVOS</td>
					</tr>
					<tr>
					<th>馆藏目录</th>
						<td>DOI.org (Crossref)</td>
					</tr>
					<tr>
					<th>URL</th>
						<td><a href="https://ieeexplore.ieee.org/document/9010040/">https://ieeexplore.ieee.org/document/9010040/</a></td>
					</tr>
					<tr>
					<th>访问时间</th>
						<td>2023/1/21 上午11:30:55</td>
					</tr>
					<tr>
					<th>地点</th>
						<td>Seoul, Korea (South)</td>
					</tr>
					<tr>
					<th>出版社</th>
						<td>IEEE</td>
					</tr>
					<tr>
					<th>ISBN</th>
						<td>978-1-72814-803-8</td>
					</tr>
					<tr>
					<th>页码</th>
						<td>8479-8488</td>
					</tr>
					<tr>
					<th>会议论文集标题</th>
						<td>2019 IEEE/CVF International Conference on Computer Vision (ICCV)</td>
					</tr>
					<tr>
					<th>会议名称</th>
						<td>2019 IEEE/CVF International Conference on Computer Vision (ICCV)</td>
					</tr>
					<tr>
					<th>DOI</th>
						<td><a href="http://doi.org/10.1109/ICCV.2019.00857">10.1109/ICCV.2019.00857</a></td>
					</tr>
					<tr>
					<th>添加日期</th>
						<td>2023/1/21 上午11:30:55</td>
					</tr>
					<tr>
					<th>修改日期</th>
						<td>2023/1/21 上午11:30:55</td>
					</tr>
				</tbody></table>
				<h3 class="attachments">附件</h3>
				<ul class="attachments">
					<li id="item_ZXMJK9K9">Duarte 等 - 2019 - CapsuleVOS Semi-Supervised Video Object Segmentat.pdf					</li>
				</ul>
			</li>


			<li id="item_EKY3FETQ" class="item journalArticle">
			<h2>Associating Objects with Transformers for Video Object Segmentation</h2>
				<table>
					<tbody><tr>
						<th>类型</th>
						<td>期刊文章</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Zongxin Yang</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Yunchao Wei</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Yi Yang</td>
					</tr>
					<tr>
					<th>摘要</th>
						<td>This paper investigates how to realize better and more 
efficient embedding learning to tackle the semi-supervised video object 
segmentation under challenging multi-object scenarios. The 
state-of-the-art methods learn to decode features with a single positive
 object and thus have to match and segment each target separately under 
multi-object scenarios, consuming multiple times computing resources. To
 solve the problem, we propose an Associating Objects with Transformers 
(AOT) approach to match and decode multiple objects uniformly. In 
detail, AOT employs an identification mechanism to associate multiple 
targets into the same high-dimensional embedding space. Thus, we can 
simultaneously process multiple objects’ matching and segmentation 
decoding as efficiently as processing a single object. For sufficiently 
modeling multi-object association, a Long Short-Term Transformer is 
designed for constructing hierarchical matching and propagation. We 
conduct extensive experiments on both multi-object and single-object 
benchmarks to examine AOT variant networks with different complexities. 
Particularly, our R50-AOT-L outperforms all the state-of-the-art 
competitors on three popular benchmarks, i.e., YouTube-VOS (84.1% J 
&amp;F), DAVIS 2017 (84.9%), and DAVIS 2016 (91.1%), while keeping more 
than 3× faster multi-object run-time. Meanwhile, our AOT-T can maintain 
real-time multi-object speed on the above benchmarks. Based on AOT, we 
ranked 1st in the 3rd Large-scale VOS Challenge.</td>
					</tr>
					<tr>
					<th>语言</th>
						<td>en</td>
					</tr>
					<tr>
					<th>馆藏目录</th>
						<td>Zotero</td>
					</tr>
					<tr>
					<th>添加日期</th>
						<td>2023/1/21 上午11:31:11</td>
					</tr>
					<tr>
					<th>修改日期</th>
						<td>2023/1/21 上午11:31:11</td>
					</tr>
				</tbody></table>
				<h3 class="attachments">附件</h3>
				<ul class="attachments">
					<li id="item_RSE7HNUM">Yang 等 - Associating Objects with Transformers for Video Ob.pdf					</li>
				</ul>
			</li>


			<li id="item_K93XK44R" class="item conferencePaper">
			<h2>A Transductive Approach for Video Object Segmentation</h2>
				<table>
					<tbody><tr>
						<th>类型</th>
						<td>会议论文</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Yizhuo Zhang</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Zhirong Wu</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Houwen Peng</td>
					</tr>
					<tr>
						<th class="author">作者</th>
						<td>Stephen Lin</td>
					</tr>
					<tr>
					<th>摘要</th>
						<td>Semi-supervised video object segmentation aims to separate a 
target object from a video sequence, given the mask in the ﬁrst frame. 
Most of current prevailing methods utilize information from additional 
modules trained in other domains like optical ﬂow and instance 
segmentation, and as a result they do not compete with other methods on 
common ground. To address this issue, we propose a simple yet strong 
transductive method, in which additional modules, datasets, and 
dedicated architectural designs are not needed. Our method takes a label
 propagation approach where pixel labels are passed forward based on 
feature similarity in an embedding space. Different from other 
propagation methods, ours diffuses temporal information in a holistic 
manner which take accounts of long-term object appearance. In addition, 
our method requires few additional computational overhead, and runs at a
 fast ∼37 fps speed. Our single model with a vanilla ResNet50 backbone 
achieves an overall score of 72.3% on the DAVIS 2017 validation set and 
63.1% on the test set. This simple yet high performing and efﬁcient 
method can serve as a solid baseline that facilitates future research. 
Code and models are available at https://github.com/ 
microsoft/transductive-vos.pytorch.</td>
					</tr>
					<tr>
					<th>日期</th>
						<td>6/2020</td>
					</tr>
					<tr>
					<th>语言</th>
						<td>en</td>
					</tr>
					<tr>
					<th>馆藏目录</th>
						<td>DOI.org (Crossref)</td>
					</tr>
					<tr>
					<th>URL</th>
						<td><a href="https://ieeexplore.ieee.org/document/9156997/">https://ieeexplore.ieee.org/document/9156997/</a></td>
					</tr>
					<tr>
					<th>访问时间</th>
						<td>2023/1/21 上午11:31:18</td>
					</tr>
					<tr>
					<th>地点</th>
						<td>Seattle, WA, USA</td>
					</tr>
					<tr>
					<th>出版社</th>
						<td>IEEE</td>
					</tr>
					<tr>
					<th>ISBN</th>
						<td>978-1-72817-168-5</td>
					</tr>
					<tr>
					<th>页码</th>
						<td>6947-6956</td>
					</tr>
					<tr>
					<th>会议论文集标题</th>
						<td>2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</td>
					</tr>
					<tr>
					<th>会议名称</th>
						<td>2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</td>
					</tr>
					<tr>
					<th>DOI</th>
						<td><a href="http://doi.org/10.1109/CVPR42600.2020.00698">10.1109/CVPR42600.2020.00698</a></td>
					</tr>
					<tr>
					<th>添加日期</th>
						<td>2023/1/21 上午11:31:18</td>
					</tr>
					<tr>
					<th>修改日期</th>
						<td>2023/1/21 上午11:31:18</td>
					</tr>
				</tbody></table>
				<h3 class="attachments">附件</h3>
				<ul class="attachments">
					<li id="item_Z5SXNLQR">Zhang 等 - 2020 - A Transductive Approach for Video Object Segmentat.pdf					</li>
				</ul>
			</li>

		</ul>
	
</body></html>

### 2.1. XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model

[Ho Kei Cheng](https://hkchengrex.github.io/), [Alexander Schwing](https://www.alexander-schwing.de/)

University of Illinois Urbana-Champaign

[[arXiv]](https://arxiv.org/abs/2207.07115) [[PDF]](https://arxiv.org/pdf/2207.07115.pdf) [[Project Page]](https://hkchengrex.github.io/XMem/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RXK5QsUo2-CnOiy5AOSjoZggPVHOPh1m?usp=sharing)

![framework](https://imgur.com/ToE2frx.jpg)

We frame Video Object Segmentation (VOS), first and foremost, as a *memory* problem.
Prior works mostly use a single type of feature memory. This can be in the form of network weights (i.e., online learning), last frame segmentation (e.g., MaskTrack), spatial hidden representation (e.g., Conv-RNN-based methods), spatial-attentional features (e.g., STM, STCN, AOT), or some sort of long-term compact features (e.g., AFB-URR).

Methods with a short memory span are not robust to changes, while those with a large memory bank are subject to a catastrophic increase in computation and GPU memory usage. Attempts at long-term attentional VOS like AFB-URR compress features eagerly as soon as they are generated, leading to a loss of feature resolution.

Our method is inspired by the Atkinson-Shiffrin human memory model, which has a *sensory memory*, a *working memory*, and a *long-term memory*. These memory stores have different temporal scales and complement each other in our memory reading mechanism. It performs well in both short-term and long-term video datasets, handling videos with more than 10,000 frames with ease.

## 