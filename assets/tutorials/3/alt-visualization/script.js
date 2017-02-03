var prefix = "/blog/assets/tutorials/3/alt-visualization/slides/slide-";
var numslides = 10;
var suffix = ".png";
  
var slideIndex = 0;
//showSlides(slideIndex); // now called in tutorial 3

function plusSlides(n) {
  showSlides(slideIndex += n);
}

function currentSlide(n) {
  showSlides(slideIndex = n);
}

function showSlides(n) {
  var img = document.getElementById("alt-visual");
  var slides = document.getElementsByClassName("mySlides");

  if (n > numslides) {slideIndex = 0}    
  if (n < 0) {slideIndex = slideIndex}
  img.src = prefix + slideIndex.toString() + suffix;
}