import numpy as np
#import matplotlib.pyplot as plt
import os
import cv2



def line_array(array):
	list_x_upper = []
	list_x_lower = []
	for y in range(5, len(array)-5):
		s_a, s_p = strtline(y, array)
		e_a, e_p = endline(y, array)
		if s_a>=7 and s_p>=5:
			list_x_upper.append(y)
			# bin_img[y][:] = 255
		if e_a>=5 and e_p>=7:
			list_x_lower.append(y)
			# bin_img[y][:] = 255
	return list_x_upper, list_x_lower


def strtline(y, array):
	count_ahead = 0
	count_prev = 0
	for i in array[y:y+10]:
		if i > 3:
			count_ahead+= 1  
	for i in array[y-10:y]:
		if i==0:
			count_prev += 1  
	return count_ahead, count_prev


def endline(y, array):
	count_ahead = 0
	count_prev = 0
	for i in array[y:y+10]:
		if i==0:
			count_ahead+= 1  
	for i in array[y-10:y]:
		if i >3:
			count_prev += 1  
	return count_ahead, count_prev


def endline_word(y, array, a):
	count_ahead = 0
	count_prev = 0
	for i in array[y:y+2*a]:
		if i < 2:
			count_ahead+= 1  
	for i in array[y-a:y]:
		if i > 2:
			count_prev += 1  
	return count_prev ,count_ahead


def end_line_array(array, a):
	list_endlines = []
	for y in range(len(array)):
		e_p, e_a = endline_word(y, array, a)
		# print(e_p, e_a)
		if e_a >= int(1.5*a) and e_p >= int(0.7*a):
			list_endlines.append(y)
	return list_endlines



def refine_endword(array):
	refine_list = []
	for y in range(len(array)-1):
		if array[y]+1 < array[y+1]:
			refine_list.append(array[y])
	refine_list.append(array[-1])
	return refine_list



def refine_array(array_upper, array_lower):
	upperlines = []
	lowerlines = []
	for y in range(len(array_upper)-1):
		if array_upper[y] + 5 < array_upper[y+1]:
			upperlines.append(array_upper[y]-10)
	for y in range(len(array_lower)-1):
		if array_lower[y] + 5 < array_lower[y+1]:
			lowerlines.append(array_lower[y]+10)

	upperlines.append(array_upper[-1]-10)
	lowerlines.append(array_lower[-1]+10)
	
	return upperlines, lowerlines



def letter_width(contours):
	letter_width_sum = 0
	count = 0
	for cnt in contours:
		if cv2.contourArea(cnt) > 20:
			x,y,w,h = cv2.boundingRect(cnt)
			letter_width_sum += w
			count += 1

	return letter_width_sum/count



def end_wrd_dtct(lines, i, bin_img, mean_lttr_width):
	count_y = np.zeros(shape = width)
	for x in range(width):
		for y in range(lines[i][0],lines[i][1]):
			if bin_img[y][x] == 255:
				count_y[x] += 1
	end_lines = end_line_array(count_y, int(mean_lttr_width))
	# print(end_lines)
	endlines = refine_endword(end_lines)
	for x in endlines:
		final_thr[lines[i][0]:lines[i][1], x] = 255
	return endlines


def start_main(src_img):
    print("\n........Program Initiated.......\n")
#    src_img= cv2.imread('/content/text.jpg',1)
    #print(src_img)
#   cv2_imshow(src_img)

    copy = src_img.copy()
    height = src_img.shape[0]
    width = src_img.shape[1]


    print("\n Resizing Image........")
    src_img = cv2.resize(copy, dsize =(1320, int(1320*height/width)), interpolation = cv2.INTER_AREA)

    height = src_img.shape[0]
    width = src_img.shape[1]

    print("#---------Image Info:--------#")
    print("\tHeight =",height,"\n\tWidth =",width)
    print("#----------------------------#")

    grey_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)


    print("Applying Adaptive Threshold with kernel :- 21 X 21")
    bin_img = cv2.adaptiveThreshold(grey_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,20)
    bin_img1 = bin_img.copy()
    bin_img2 = bin_img.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    kernel1 = np.array([[1,0,1],[0,1,0],[1,0,1]], dtype = np.uint8)
    # final_thr = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    # final_thr = cv2.dilate(bin_img,kernel1,iterations = 1)

    print("Noise Removal From Image.........")
    final_thr = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    contr_retrival = final_thr.copy()


    print("Beginning Character Semenation..............")
    count_x = np.zeros(shape= (height))
    for y in range(height):
        for x in range(width):
            if bin_img[y][x] == 255 :
                count_x[y] = count_x[y]+1
        # print(count_x[y])

    # t = np.arange(0,height, 1)
    # plt.plot(t, count_x[t])
    # plt.axis([0, height, 0, 350])

    upper_lines, lower_lines = line_array(count_x)

    upperlines, lowerlines = refine_array(upper_lines, lower_lines)

    # print(upperlines, lowerlines)
    if len(upperlines)==len(lowerlines):
        lines = []
        for y in upperlines:
            final_thr[y][:] = 255	
        for y in lowerlines:
            final_thr[y][:] = 255
        for y in range(len(upperlines)):
            lines.append((upperlines[y], lowerlines[y]))
        
    else:
        print("Too much noise in image, unable to process.\nPlease try with another image. Ctrl-C to exit:- ")
        #showimages()
        #k = cv2.waitKey(0)
        #while 1:
        #	k = cv2.waitKey(0)
        #	if k & 0xFF == ord('q'):
        #		cv2.destroyAllWindows()

    lines = np.array(lines)

    no_of_lines = len(lines)

    print("\nGiven Text has   # ",no_of_lines, " #   no. of lines")

    lines_img = []

    for i in range(no_of_lines):
        lines_img.append(bin_img2[lines[i][0]:lines[i][1], :])

    contours, hierarchy = cv2.findContours(contr_retrival,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    final_contr = np.zeros((final_thr.shape[0],final_thr.shape[1],3), dtype = np.uint8)
    cv2.drawContours(src_img, contours, -1, (0,255,0), 1)

    mean_lttr_width = letter_width(contours)
    print("\nAverage Width of Each Letter:- ", mean_lttr_width)
    mean_lttr_width=9 ############# this i have done for testing and added manually original was 19.85 and gives better result

    x_lines = [] # these are for word detection in the line

    for i in range(len(lines_img)):
        x_lines.append(end_wrd_dtct(lines, i, bin_img, mean_lttr_width))

    for i in range(len(x_lines)):
        x_lines[i].append(width)

    print(x_lines)
            
    return lines, lines_img, x_lines
