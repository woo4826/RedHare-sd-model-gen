import requests,logging
import asyncio


async def send_get_request(key):
    # 외부 서버의 URL 설정
    external_url = f'http://localhost:4000/train/mdlalnzjvh'

    try:
        # GET 요청 보내기
        logging.info(external_url)
        response = requests.get(external_url)

        # 응답 확인
        if response.status_code == 200:
            logging.info(response.text)
            return f'Success! Response: {response.text}'
        else:
            logging.info(response.status_code)
            return f'Error! Status code: {response.status_code}'

    except Exception as e:
        return f'An error occurred: {str(e)}'


asyncio.run(send_get_request("d"))

