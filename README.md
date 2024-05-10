
# RedHare-sd-model-gen

소개
------
stable diffusion에서 활용할 수 있는 LoRA 자동생성 
- 학습할 이미지를 전송하면 LoRA를 생성해준다.


주요기능
-----
1. 이미지 태그 생성
2. LoRA 학습 및 생성


실행방법
----
실행
- docker compose up

LoRA생성 요청
- post
  - /model
- body
  - user_id : string
  - independant_key : string
  - files : images(jpg or png)
 
생성완료
- 생성 위치: workspace/output
