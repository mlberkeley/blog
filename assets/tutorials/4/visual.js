// dimensions of each frame
var imgWidth = 400;
var imgHeight = 300;

// small screen size upper limit
var smallScreen = 700; 
// 700 appears to be upper limit for phones, lower limit for tablets

if (window.innerWidth < smallScreen) {
  imgWidth = imgWidth / 2;
  imgHeight = imgHeight / 2;
} 
// dimensions of image in frames
var rows = 24;
var cols = 5;

var topOffset = -30;

var startVal = +document.getElementById("image_id").getAttribute("start");
showVal(startVal);
function showVal(newVal){
    document.getElementById("test").innerHTML="Complexity: " + newVal / 100;
    showImage(newVal);
}

function showImage(newVal) {
  var pred = document.getElementById("predBox");
  var err = document.getElementById("errBox");

    pred.style.width = imgWidth.toString() + "px";
    pred.style.height = imgHeight.toString() + "px";


    err.style.width = imgWidth.toString() + "px";
    err.style.height = imgHeight.toString() + "px";
  
  var predImg = document.getElementById("predPic");
  predImg.style.top = ((-1 * (+newVal) * imgHeight) % (rows * imgHeight)).toString() + "px";
  predImg.style.left = ((-1 * parseInt(+newVal / rows) * imgWidth) ).toString() + "px";
  
  var errImg = document.getElementById("errPic");
  errImg.style.top = ((-1 * (+newVal) * imgHeight) % (rows * imgHeight)).toString() + "px";
  errImg.style.left = ((-1 * parseInt(+newVal / rows) * imgWidth) ).toString() + "px";
  

  if (window.innerWidth < smallScreen) {
    predImg.style.width = "1000px";
    
    errImg.style.width = "1000px";
  }
}
