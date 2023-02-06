ABCD(Any Body Can be a Designer)
---

Last modified: 2023.02.05

<p align="center"><img src="https://user-images.githubusercontent.com/75806377/216806082-164a0d58-d314-4f34-a17e-4f82bd518e77.png" height="300px" width="500px"></p>  

ABCD는 'Any Body Can be a Designer'의 약자로 통합 그래픽 작업 플랫폼입니다.  
학부생 때 단독으로 진행했던 졸업프로젝트이며 누구나 쉽게 누끼를 따면서 어려운 그래픽 작업을 쉽게하고  
또 그림을 자동으로 생성해주는 neural style transfer 등을 통해 개인화된 경험을 제공하고자 하였습니다.

기능은 총 4가지로 구성되어 있습니다.  
- Image classification and Providing the abstract image
- Create a new sketch
- Remove a background
- Neural style transfer  

4가지의 기능을 모두 포함한 ABCD 플랫폼은 streamlit으로 작업하였으며  
Image classification and Providing the abstract image 부분에서는  
input image의 class를 예측하여 그에 해당하는 abstract image(Illust, Sketch, Pictogram)을 제공하는 기능입니다.  
abstract image는 [AI-HUB](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=617)에서 다운로드가 가능하나 저작권 침해 문제가 발생할 수 있기에  
만약 활용하신다면 직접 다운받으신 후 local 환경에서 구동하시길 바랍니다.  

</br>

Image classification and Providing the abstract image
---
<p align='center'><img src="https://user-images.githubusercontent.com/75806377/216807004-96f25ffc-dcff-4d82-95c7-e22586b2fdd8.jpg" height="200px" width="300px"></p>  

혹시 조정 경기에서 연습용 기구로 쓰이는 해당 기구의 정식 명칭을 알고 계신가요..?  
ABCD에 구현한 Image classification and Providing the abstract image 는 저 그림에서 출발하였는데요.  
해당 기구의 이름은 '로잉 머신(Rowing machine)'인데 Google에 검색하고자 키보드에 손을 올려놓는 순간  
'아..그거 뭐였지.. 왔다갔다하는 기구..' 이러면서 로잉 머신을 찾는데 별의 별 짓을 다 했었습니다..  

Image classification and Providing the abstract image는 제가 직면하였던 문제점을 해결하고자 시작되었습니다.  
Image를 upload할 경우 해당하는 class를 예측하는 단순 prediction하는 기능입니다.  

Prediction 후에 class에 대한 Illust, Skecth, Pictogram을 제공하며    
Image classification을 위해 Resnet-50 model을 활용하였고 해당하는 class에 대한 Illust, Skecth, Pictogram는  
expander로 구성하여 원하는 메뉴를 클릭하여 다운로드할 수 있도록 구성하였습니다.  
제가 사용한 데이터셋의 class는 영문명이기에 eng_class를 kor_class로 해석할 수 있도록 class_dict를 따로 구성하였습니다.  
</br>
</br>

![최종발표자료_복사본-008 (1)](https://user-images.githubusercontent.com/75806377/216962184-fc0d6983-4027-4dc7-baa6-110d8b54128e.png)

</br>

Create a new sketch
---
Create a new sketch는 말 그대로 Sketch를 만드는 기능입니다.  
사전에 배경을 먼저 제거하여 배경 속 불필요한 요소들까지 edge가 검출되지 않도록 구성하였는데요.  
요건 배경을 제거하는 기능을 별도로 분리하였기에 자세한 설명은 뒤에서 설명드리겠습니다.  
Image 속 다양한 요소들의 edge는 픽셀값이 급격히 변하는 부분이기에 미분을 통해 edge를 검출할 수 있습니다.  
원하는 그림에 대한 sketch를 얻고 싶을 때 해당 기능이 유용할 수 있는데요.  

기존 edge detection을 위한 다양한 kernel들이 많이 존재합니다.  
Robert filter, Sobel filter, Prewitt filter 등 다양한 HPF(High-Pass Filter)들이 존재하는데  
저는 다양한 HPF와의 convolution 결과값들을 모두 사용하는 방향으로 최종 sketch를 제작하였습니다.  

굳이 한 HPF와의 convolution한 값을 최종 sketch로 선택하지 않고 모든 결과값들을 사용하는 이유는  
sktech는 어느 부분은 선의 굵기가 얇고 어느 부분은 굵은 즉, 자연스러운 명암과 같은 효과를 주기 위함입니다.  
만약 5개의 결과값 중에서 4개의 결과값은 edge가 없다고 결과값이 나왔지만 나머지 1개의 결과값에선  
값이 작더라도 미세하게나마 edge를 detect했을 경우 작게나마 그러한 것들을 표현하기 위함인데요.  
저는 이러한 목표를 달성하기 위해 다양한 kernel들과 convolution한 결과값 중 max값을 기반으로  
최종 sketch를 제작하는 방식으로 보다 자연스러운 느낌의 sketch를 제작하였습니다.  
</br>
</br>

![최종발표자료-009](https://user-images.githubusercontent.com/75806377/216808074-a2068d44-6d5c-4eed-8efa-fa82c719e6cf.png)

</br>

Remove background
---
Remove background는 배경을 제거하는 즉, 누끼를 따는 작업입니다.  
앞서 Create a new sketch 때도 Remove background function이 적용되었습니다.  

해당 기능은 현재까지 많이 사용되며 Semantic segmentation의 목표와 부합합니다.  
Instance와 Background를 분리하기 위함이 목적이며 보다 매끄럽고 확실한 segmentation을 위해  
저는 U-Net을 기반으로 segmentation을 진행하였습니다.  

U-Net은 expanding path와 정확한 localization을 위한 contracting path가 'U'자의 형태로 구성되어 있는 model이며  
pixel-wise classification을 진행한 후 overlap-tile strategy, mirroring extrapolate, weight loss 등을 사용하여  
배경을 제거하였습니다.  
</br>
~~(고양이, 썬더람쥐, 시바 너무 귀엽습니다..)~~

</br>

![최종발표자료-011](https://user-images.githubusercontent.com/75806377/216808456-59890fa9-5a2c-481c-bc1c-77428897b08d.png)

</br>

Neural Style Transfer
---


Neural Style Transfer는 Leon A. Gatys의 논문 'A Neural Algorithm of Artistic Style(2015)'에서  
소개된 개념이며 화가의 화풍을 기반으로 그림을 재구성하는 알고리즘입니다.  
해당 논문을 보면서 ABCD 플랫폼 안에서 '나도 그림을 만들 수 있다'와 같이 개인화된 경험을 제공하고자  
기능을 실제로 구현해봤으며 특정 과거의 유명했던 화가의 화풍이 아닐지라도 본인이 원하는 화풍을  
upload하여 만들어 볼 수 있다는 점에서 매력적인 알고리즘을 ABCD에서 실제로 구현해보았습니다.  

Neural style transfer는 VGG-19를 기반으로 하고 있으며  
보다 쉽게 이해하자면 '화풍의 느낌을 살려 그림을 변화시켜보자' 입니다.  

우선 random noise로 empty canvas를 만든 후 Random noise(결과값), Style image(화풍),  
Content image(변화시키고자 하는 이미지) 총 3개를 각각 VGG-19 기반의 신경망에 convolution합니다.  
이 때, 중요한 점은 화풍의 느낌을 살린 이미지이기에 content image의 detail한 요소들은 별로 중요하지 않게됩니다.  
그래서 content image의 경우 detatil한 요소들이 살아있는 앞 layer 쪽의 convolution 결과값보다  
반복적인 convolution을 통해 많이 뭉개져버린 뒷 layer 쪽의 convolution 결과값을 채택하여 random noise와의 MSE 값을 줄입니다.  
Style image의 경우는 random하게 convolution layer 5개 정도를 선택하여 [Gram matrix](https://wansook0316.github.io/ds/dl/2020/09/18/computer-vision-15-Gram-Matrix.html)를 계산한 뒤 합산된 MSE 값을 줄이며  
random noise는 반복적으로 수행하면서 화풍과 변화시키고자 하는 이미지 그 중간의 이미지로 향하게 됩니다.  

</br>

![최종발표자료-012](https://user-images.githubusercontent.com/75806377/216809383-cc9b5fa5-5618-4600-bbc0-eea9035cc815.png)
![image](https://user-images.githubusercontent.com/75806377/216960293-6cb2c7f4-5ffb-4842-bcbf-84f868af2af8.png)
</br>

HOW TO USE?
---

- Install packages
```Python
pip install -r requirements.txt 
```

- path 수정  
만약 local 환경에서 구동하시길 원하신다면 [1_Upload_and_Searching.py] 와 [5_Archive.py] 에서  
데이터를 import하는 부분의 path를 다운받으신 경로에 맞게 수정해주시면 됩니다.

- Tensorflow hub 문제 발생시  
[4_Neural_style_transfer.py]에서 saved_model.pb가 없다고 오류가 뜰 경우  
magenta_arbitrary-image-stylization-v1-256_2 를 tensorflow hub에서 다운받으신 후  
없다고 뜬 경로에 맞게 copy&paste 하시면 됩니다.

```Python
streamlit run home.py
```
