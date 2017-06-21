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
  var div = document.getElementById("predBox");
  div.style.width = imgWidth.toString() + "px";
  div.style.height = imgHeight.toString() + "px";
  
  var div = document.getElementById("errBox");
  div.style.width = imgWidth.toString() + "px";
  div.style.height = imgHeight.toString() + "px";
  
  var img = document.getElementById("predPic");
  img.style.top = ((-1 * (+newVal) * imgHeight) % (rows * imgHeight)).toString() + "px";
  img.style.left = ((-1 * parseInt(+newVal / rows) * imgWidth) ).toString() + "px";
  
  var img = document.getElementById("errPic");
  img.style.top = ((-1 * (+newVal) * imgHeight) % (rows * imgHeight)).toString() + "px";
  img.style.left = ((-1 * parseInt(+newVal / rows) * imgWidth) ).toString() + "px";
}