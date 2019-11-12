import aiohttp
import asyncio
import uvicorn
import test
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://www.googleapis.com/drive/v3/files/1-4Bszwv0HDTmHd2LEQmDl3KfMP0z9PuC?alt=media&key=AIzaSyBg-HoeVUlHZNf5YgAtPmhDFNrVnAD-WuQ'
export_file_name = 'export.pkl'

classes = ['Sample001','Sample002','Sample003','Sample004','Sample005','Sample006','Sample007','Sample008','Sample009','Sample010','Sample011','Sample012','Sample013','Sample014','Sample015','Sample016','Sample017','Sample018','Sample019','Sample020','Sample021','Sample022','Sample023','Sample024','Sample025','Sample026','Sample027','Sample028','Sample029','Sample030','Sample031','Sample032','Sample033','Sample034','Sample035','Sample036','Sample037','Sample038','Sample039','Sample040','Sample041','Sample042','Sample043','Sample044','Sample045','Sample046','Sample047','Sample048','Sample049','Sample050','Sample051','Sample052','Sample053','Sample054','Sample055','Sample056','Sample057','Sample058','Sample059','Sample060','Sample061','Sample062']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    test_n = test.test()
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
