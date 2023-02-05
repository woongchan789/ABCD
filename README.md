ABCD(Any Body Can be a Designer)
---
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
abstract image는 AI-HUB([Data](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=617))에서 다운로드가 가능하나 저작권 침해 문제가 발생할 수 있기에  
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
Image classification을 위해 Resnet-101 model을 활용하였고 해당하는 class에 대한 Illust, Skecth, Pictogram는  
expander로 구성하여 원하는 메뉴를 클릭하여 다운로드할 수 있도록 구성하였습니다.  
제가 사용한 데이터셋의 class는 영문명이기에 eng_class를 kor_class로 해석할 수 있도록 class_dict를 따로 구성하였습니다.  
</br>
</br>
![최종발표자료-007](https://user-images.githubusercontent.com/75806377/216807627-3bbdc363-aa41-4833-bfe0-4719b2ceeb92.png)

</br>

Create a new sketch
---
Create a new sketch는 말 그대로 Sketch를 만드는 기능입니다.  
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
