from flask import Flask, render_template, request
from skimage.color import rgb2lab, lab2rgb
from PIL import Image


import numpy as np
import cv2
import shutil
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

def bgr2rgb(img):
    b,g,r = cv.split(img)
    return cv.merge([b,g,r])
    
    
def get_dark_channel_prior(img, w_size):
    """
    img    -> 3D tensor in RGB format
    w_size -> size of patch to consider (default is 15)
    """    
    J_dark = ndimage.minimum_filter(img, footprint=np.ones((w_size,w_size,3)), mode='nearest')
            
    return J_dark[:,:,1]
    
def estimate_atmospheric_light(img, w_size):
    """
    img -> 3D tensor in RGB format
    
    ret -> 
        A_r |
        A_g | -> estimated atmospheric light in the RGB channels
        A_c |
    """
    size = img.shape[:2]
    k = int(0.001*np.prod(size))
    j_dark = get_dark_channel_prior(img, w_size=w_size)
    idx = np.argpartition(-j_dark.ravel(),k)[:k]
    x, y = np.hsplit(np.column_stack(np.unravel_index(idx, size)), 2)
    
    A = np.array([img[x,y,0].max(), img[x,y,1].max(), img[x,y,2].max()])
    return A
    

    
def estimate_transmission(img, w_size, omega=0.95 ):
    """
    Estimates the transmission map using the dark channel prior of the normalized image. 
    A small fraction, omega, of the haze is kept to retain depth perspective after haze removal.
    
    img   -> 3D Tensor in RGB format
    omega -> fraction of haze to keep in image (default is 0.95)
    """
    A= estimate_atmospheric_light(img,w_size)
    norm_img = img / A
    norm_img_dc = get_dark_channel_prior(norm_img, w_size=w_size)

    return 1 - omega*norm_img_dc
    
def guided_filter(I, p, omega=60, eps=0.01):
    """
    from http://kaiminghe.com/publications/eccv10guidedfilter.pdf
    and  https://arxiv.org/pdf/1505.00996.pdf
    
    I     -> guidance image, 3D Tensor in RGB format
    p     -> filtering input image, 
    omega -> window size (default is 60)
    eps   -> regularization parameter (default 0.01)
    """
    
    w_size = (omega,omega)
    I = I/255
    I_r, I_g, I_b = I[:,:,0], I[:,:,1], I[:,:,2]
    
    mean_I_r = cv.blur(I_r, w_size)
    mean_I_g = cv.blur(I_g, w_size)
    mean_I_b = cv.blur(I_b, w_size)
    
    mean_p = cv.blur(p, w_size)
    
    mean_Ip_r = cv.blur(I_r*p, w_size)
    mean_Ip_g = cv.blur(I_g*p, w_size)
    mean_Ip_b = cv.blur(I_b*p, w_size)
         
    cov_Ip_r =  mean_Ip_r - mean_I_r*mean_p
    cov_Ip_g =  mean_Ip_g - mean_I_g*mean_p
    cov_Ip_b =  mean_Ip_b - mean_I_b*mean_p
    cov_Ip = np.stack([cov_Ip_r, cov_Ip_g, cov_Ip_b], axis=-1)
    
    var_I_rr = cv.blur(I_r*I_r, w_size) - mean_I_r*mean_I_r
    var_I_rg = cv.blur(I_r*I_g, w_size) - mean_I_r*mean_I_g
    var_I_rb = cv.blur(I_r*I_b, w_size) - mean_I_r*mean_I_b
    var_I_gb = cv.blur(I_g*I_b, w_size) - mean_I_g*mean_I_b
    var_I_gg = cv.blur(I_g*I_g, w_size) - mean_I_g*mean_I_g
    var_I_bb = cv.blur(I_b*I_b, w_size) - mean_I_b*mean_I_b
    
    a = np.zeros(I.shape)
    for x, y in np.ndindex(I.shape[:2]):
        Sigma = np.array([
            [var_I_rr[x,y], var_I_rg[x,y], var_I_rb[x,y]],
            [var_I_rg[x,y], var_I_gg[x,y], var_I_gb[x,y]],
            [var_I_rb[x,y], var_I_gb[x,y], var_I_bb[x,y]]
        ])
        c = cov_Ip[x,y,:]
        
        a[x,y,:] = np.linalg.inv(Sigma + eps*np.eye(3)).dot(c)
        
    mean_a = np.stack([cv.blur(a[:,:,0], w_size), cv.blur(a[:,:,1], w_size), cv.blur(a[:,:,2], w_size)], axis=-1)
    mean_I = np.stack([mean_I_r, mean_I_g, mean_I_b], axis=-1)
    
    b = mean_p - np.sum(a*mean_I, axis=2)
    mean_b = cv.blur(b, w_size)
    q = np.sum(mean_a*I, axis=2) + mean_b
    
    return q
    
def haze_removal(img, w_size, a_omega=0.95, gf_w_size=200, eps=1e-6):
    """
    Implements the haze removal pipeline from 
    Single Image Haze Removal Using Dark Channel Prior by He et al. (2009)
    
    I       -> 3D tensor in RGB format
    w_size  -> window size of local patch (default is 15)
    a_omega -> fraction of haze to keep in image (default is 0.95)
    omega   -> window size for guided filter (default is 200)
    eps     -> regularization parameter for guided filter(default 1e-6)
    """
    img = img.astype(np.int16)
    A = estimate_atmospheric_light(img, w_size=w_size)
    alpha_map = estimate_transmission(img, omega=a_omega, w_size=w_size)
    f_alpha_map = guided_filter(img, alpha_map, omega=gf_w_size, eps=eps)
    
    img[:,:,0] -= A[0]
    img[:,:,1] -= A[1]
    img[:,:,2] -= A[2]
    z = np.maximum(f_alpha_map, 0.1)
    img[:,:,0] = img[:,:,0]/z
    img[:,:,1] = img[:,:,1]/z
    img[:,:,2] = img[:,:,2]/z

    img[:,:,0] += A[0]
    img[:,:,1] += A[1]
    img[:,:,2] += A[2]

    img = np.maximum(img, 0)
    img = np.minimum(img, 255)
    
    return img, f_alpha_map
    


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/')
def index():
	return render_template("index.html")

@app.route('/comparison')
def index1():
	return render_template("second.html")
@app.route('/comparison1')
def index2():
	return render_template("third.html")
@app.route("/submit", methods=["POST"])
def prediction():
    img1_color=[]
    img=request.files['img']
    img_path1 = 'static/haze.jpg'
    img.save(img_path1)
    img = bgr2rgb(cv.imread("static/haze.jpg"))
    w_size1 = int(request.form["param1"])
    #cv2.imwrite(path,img_to_save)
    cv.imwrite("static/original_img.jpg",img)
    img2 = get_dark_channel_prior(img, w_size1)
    img4 = estimate_atmospheric_light(img, w_size1)
    img5 = estimate_transmission(img,w_size1, omega=0.95)
    
    l, _ = haze_removal(img, w_size1, a_omega=0.95, gf_w_size=200, eps=1e-6)
    #f, ax2= plt.subplots(1, 1, figsize=(10,10))
    cv.imwrite("static/final_img1.jpg",img2)
    cv.imwrite("static/final_img2.jpg",img4)
    cv.imwrite("static/final_img4.jpg",img5)
    cv.imwrite("static/final_img.jpg",l)

    return render_template("index.html", img_path1='static/original_img.jpg', img_path2='static/final_img1.jpg', img_path3 =  'static/final_img2.jpg', img_path4 =  'static/final_img4.jpg', img_path5 =  'static/final_img.jpg')
@app.route("/sub", methods=["POST"])
def prediction1():
    img1_color=[]
    img=request.files['img']
    img_path1 = 'static/haze.jpg'
    img.save(img_path1)
    img = bgr2rgb(cv.imread("static/haze.jpg"))
    w_size1=1
    #cv2.imwrite(path,img_to_save)
    cv.imwrite("static/original_img.jpg",img)
    img12 = get_dark_channel_prior(img, w_size1)
   
    
    l1, _ = haze_removal(img, w_size1, a_omega=0.95, gf_w_size=200, eps=1e-6)
    w_size1=3
    img32 = get_dark_channel_prior(img, w_size1)
  
    l3, _ = haze_removal(img, w_size1, a_omega=0.95, gf_w_size=200, eps=1e-6)
    w_size1=5
    img52 = get_dark_channel_prior(img, w_size1)

    
    l5, _ = haze_removal(img, w_size1, a_omega=0.95, gf_w_size=200, eps=1e-6)
    w_size1=7
    img72 = get_dark_channel_prior(img, w_size1)
   
    
    l7, _ = haze_removal(img, w_size1, a_omega=0.95, gf_w_size=200, eps=1e-6)
    w_size1=9
    img92 = get_dark_channel_prior(img, w_size1)
   
    
    l9, _ = haze_removal(img, w_size1, a_omega=0.95, gf_w_size=200, eps=1e-6)
    w_size1=13
    img132 = get_dark_channel_prior(img, w_size1)
   
    
    l13, _ = haze_removal(img, w_size1, a_omega=0.95, gf_w_size=200, eps=1e-6)
    w_size1=15
    img152 = get_dark_channel_prior(img, w_size1)
  
    print(w_size1)
    l15, _ = haze_removal(img, w_size1, a_omega=0.95, gf_w_size=200, eps=1e-6)
    #f, ax2= plt.subplots(1, 1, figsize=(10,10))
    cv.imwrite("static/final_img11.jpg",img12)
    
    cv.imwrite("static/final_img1.jpg",l1)
    #3
    cv.imwrite("static/final_img31.jpg",img32)
    
    cv.imwrite("static/final_img3.jpg",l3)

    #5
    cv.imwrite("static/final_img51.jpg",img52)
    
    cv.imwrite("static/final_img5.jpg",l5)
    #7
    cv.imwrite("static/final_img71.jpg",img72)
 
    cv.imwrite("static/final_img7.jpg",l7)

    cv.imwrite("static/final_img91.jpg",img92)

    cv.imwrite("static/final_img9.jpg",l9)

    cv.imwrite("static/final_img131.jpg",img132)
   
    cv.imwrite("static/final_img13.jpg",l13)

    cv.imwrite("static/final_img151.jpg",img152)
   
    cv.imwrite("static/final_img15.jpg",l15)

    return render_template("second.html", img_path1='static/original_img.jpg', img_path11='static/final_img11.jpg', img_path15 =  'static/final_img1.jpg',
    img_path31='static/final_img31.jpg', img_path35 =  'static/final_img3.jpg',
    img_path51='static/final_img51.jpg',  img_path55 =  'static/final_img5.jpg',
    img_path71='static/final_img71.jpg',  img_path75 =  'static/final_img7.jpg',
    img_path91='static/final_img91.jpg', img_path95 =  'static/final_img9.jpg',
    img_path131='static/final_img131.jpg',  img_path135 =  'static/final_img13.jpg',
    img_path151='static/final_img151.jpg',  img_path155 =  'static/final_img15.jpg'
    )
@app.route("/siuuu", methods=["POST"])
def prediction3():
    img1_color=[]
    img=request.files['img']
    img_path1 = 'static/haze.jpg'
    img.save(img_path1)
    img = bgr2rgb(cv.imread("static/haze.jpg"))
    w_size1 = int(request.form["param1"])
    #cv2.imwrite(path,img_to_save)
    cv.imwrite("static/original_img.jpg",img)
    img26 = get_dark_channel_prior(img, w_size1)
    img46 = estimate_atmospheric_light(img, w_size1)
    img56 = estimate_transmission(img,w_size1, omega=0.95)
    
    l6, _ = haze_removal(img, w_size1, a_omega=0.95, gf_w_size=200, eps=1e-6)
    #f, ax2= plt.subplots(1, 1, figsize=(10,10))
    cv.imwrite("static/final_img1.jpg",img26)
    cv.imwrite("static/final_img2.jpg",img46)
    cv.imwrite("static/final_img4.jpg",img56)
    cv.imwrite("static/final_img.jpg",l6)

    return render_template("third.html", img_path1='static/original_img.jpg', img_path2='static/final_img1.jpg', img_path3 =  'static/final_img2.jpg', img_path4 =  'static/final_img4.jpg', img_path5 =  'static/final_img.jpg')
if __name__ == "__main__":
	app.run()
