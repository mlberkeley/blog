var imgWidth = 600;
var imgHeight = 400;

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
  img.style.top = "-30px";
  img.style.left = (-1 * (+newVal) * imgWidth).toString() + "px";
  
  var img = document.getElementById("errPic");
  img.style.top = "-30px";
  img.style.left = (-1 * (+newVal) * imgWidth).toString() + "px";
}