import aiohttp
import asyncio
import uvicorn
import cv2
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from char_seg import *


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


def letter_seg(lines_img, x_lines, i):
	import time
	copy_img = lines_img[i].copy()
	x_linescopy = x_lines[i].copy()
	
	letter_img = []
	letter_k = []
	
	x, contours, hierarchy = cv2.findContours(copy_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)	
	for cnt in contours:
		if cv2.contourArea(cnt) > 50:
			x,y,w,h = cv2.boundingRect(cnt)
			# letter_img.append(lines_img[i][y:y+h, x:x+w])
			letter_k.append((x,y,w,h))
	#print(letter_k)
	letter = sorted(letter_k, key=lambda student: student[0])
	#print(letter)
	# print(letter)
	mapping={
		"Sample001" : "0",
		"Sample002" : "1",
		"Sample003" : "2",
		"Sample004" : "3",
		"Sample005" : "4",
		"Sample006" : "5",
		"Sample007" : "6",
		"Sample008" : "7",
		"Sample009" : "8",
		"Sample010" : "9",
		"Sample011" : "A",
		"Sample012" : "B",
		"Sample013" : "C",
		"Sample014" : "D",
		"Sample015" : "E",
		"Sample016" : "F",
		"Sample017" : "G",
		"Sample018" : "H",
		"Sample019" : "I",
		"Sample020" : "J",
		"Sample021" : "K",
		"Sample022" : "L",
		"Sample023" : "M",
		"Sample024" : "N",
		"Sample025" : "O",
		"Sample026" : "P",
		"Sample027" : "Q",
		"Sample028" : "R",
		"Sample029" : "S",
		"Sample030" : "T",
		"Sample031" : "U",
		"Sample032" : "V",
		"Sample033" : "W",
		"Sample034" : "X",
		"Sample035" : "Y",
		"Sample036" : "Z",
		"Sample037" : "a",
		"Sample038" : "b",
		"Sample039" : "c",
		"Sample040" : "d",
		"Sample041" : "e",
		"Sample042" : "f",
		"Sample043" : "g",
		"Sample044" : "h",
		"Sample045" : "i",
		"Sample046" : "j",
		"Sample047" : "k",
		"Sample048" : "l",
		"Sample049" : "m",
		"Sample050" : "n",
		"Sample051" : "o",
		"Sample052" : "p",
		"Sample053" : "q",
		"Sample054" : "r",
		"Sample055" : "s",
		"Sample056" : "t",
		"Sample057" : "u",
		"Sample058" : "v",
		"Sample059" : "w",
		"Sample060" : "x",
		"Sample061" : "y",
		"Sample062" : "z"
}
	word = 1
	letter_index = 0
	#to show line
	cv2_imshow(lines_img[i][letter[0][1]-5:letter[len(letter)-1][1]+letter[len(letter)-1][3]+5,letter[0][0]-5:letter[len(letter)-1][0]+letter[len(letter)-1][2]+5])
	#cv2_imshow(lines_img[i][letter[0][1]-5:letter[len(letter)-1][1]+letter[len(letter)-1][3]+5,x_linescopy[0]-5:letter[len(letter)-1][0]+letter[len(letter)-1][2]+5])
	line_of_word=''
	for e in range(len(letter)):
		if(letter[e][0]<x_linescopy[0]):
			letter_index += 1
			letter_img_tmp = lines_img[i][letter[e][1]-5:letter[e][1]+letter[e][3]+5,letter[e][0]-5:letter[e][0]+letter[e][2]+5]
			letter_img = cv2.resize(letter_img_tmp, dsize =(124, 124), interpolation = cv2.INTER_AREA)
			White = [0,0]
			letter_img= cv2.copyMakeBorder(letter_img.copy(),50,50,50,50,cv2.BORDER_CONSTANT,value=White)
			cv2.imwrite('check.jpg', 255-letter_img)
			time.sleep(0.1)
			img = open_image('check.jpg')
			pred_class,pred_idx,outputs = learn.predict(img)
			#cv2_imshow(255-letter_img)
			#print(pred_class)
			#print(mapping[str(pred_class)])
			line_of_word=line_of_word+mapping[str(pred_class)]
			#learn.predict(255-letter_img)
		else:
			x_linescopy.pop(0)
			word += 1
			letter_index = 1
			letter_img_tmp = lines_img[i][letter[e][1]-5:letter[e][1]+letter[e][3]+5,letter[e][0]-5:letter[e][0]+letter[e][2]+5]
			letter_img = cv2.resize(letter_img_tmp, dsize =(124, 124), interpolation = cv2.INTER_AREA)
			White = [0,0]
			letter_img= cv2.copyMakeBorder(letter_img.copy(),50,50,50,50,cv2.BORDER_CONSTANT,value=White)
			cv2.imwrite('check.jpg', 255-letter_img)
			time.sleep(0.1)
			img = open_image('check.jpg')
			pred_class,pred_idx,outputs = learn.predict(img)
			#cv2_imshow(255-letter_img)
			#print(pred_class)
			#print(mapping[str(pred_class)])
			line_of_word=line_of_word+' '+mapping[str(pred_class)]
			# print(letter[e][0],x_linescopy[0], word)

	return line_of_word				



@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    print(type(img))
    print(img)
#    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #cv2.imwrite('new.jpg',gray)
    #img2 = cv2.imread('new.jpg',1) 
#    lines, lines_img, x_lines = start_main(gray)
    final_out = ''
#    for i in range(len(lines)):    # i is the line number
#	    final_out = final_out + letter_seg(lines_img, x_lines, i)	#all
	    #print(i)
    #prediction = learn.predict(img)[0]
    return JSONResponse({'result': final_out+type(img)+' '+str(img)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
