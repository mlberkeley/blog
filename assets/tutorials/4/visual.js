

window.addEventListener('load', function() {
  if (window.innerWidth < 1000) {
    
  }
})
// dimensions of each frame
var imgWidth = 400;
var imgHeight = 300;

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
  
  if (window.innerWidth < 1000) {
    pred.style.width = (imgWidth / 2).toString() + "px";
    pred.style.height = (imgHeight / 2).toString() + "px";
    
    err.style.width = (imgWidth / 2).toString() + "px";
    err.style.height = (imgHeight / 2).toString() + "px";
  } else {
    pred.style.width = imgWidth.toString() + "px";
    pred.style.height = imgHeight.toString() + "px";


    err.style.width = imgWidth.toString() + "px";
    err.style.height = imgHeight.toString() + "px";
  }
  
  var predImg = document.getElementById("predPic");
  predImg.style.top = ((-1 * (+newVal) * imgHeight) % (rows * imgHeight)).toString() + "px";
  predImg.style.left = ((-1 * parseInt(+newVal / rows) * imgWidth) ).toString() + "px";
  
  var errImg = document.getElementById("errPic");
  errImg.style.top = ((-1 * (+newVal) * imgHeight) % (rows * imgHeight)).toString() + "px";
  errImg.style.left = ((-1 * parseInt(+newVal / rows) * imgWidth) ).toString() + "px";
  
  if (window.innerWidth < 1000) {
    predImg.style.width = "1000px";
    predImg.style.left = ((-1 * parseInt(+newVal / rows) * imgWidth) / 2).toString() + "px";
    
    errImg.style.width = "1000px";
    errImg.style.left = ((-1 * parseInt(+newVal / rows) * imgWidth) / 2).toString() + "px";
  }
}