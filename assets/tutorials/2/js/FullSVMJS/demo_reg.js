
/*
This demo includes npg library (notpygamejs) that I wrote a while ago. It can
be found on my Github page (https://github.com/karpathy/notpygamejs). It's a 
quick and dirty canvas wrapper that has some helped functions I use often.
It's main use is to contain a main loop and expose methods update(), draw(), 
as well as some events keyUp(), mouseClick() etc.
*/


var N= 2; //number of data points
var data = new Array(N);
var labels= new Array(N);
var wb; // weights and offset structure
var ss= 50.0; // scaling factor for drawing
var svm= new svmjs.SVM();
var trainstats;
var dirty= true;
var kernelid= 0;
var rbfKernelSigma = 0.5;
var svmC = 1.0;
var selected = -1;
var polyD = 2;
var polyC = 0;

function myinit(){
  
  N = 2;
  data = new Array(N);
  labels = new Array(N);
  data[0]=[0  ,  1 ];
  data[1]= [1, 0];
  labels[0]= 1;
  labels[1]= -1;
  
  retrainSVM();
}

function retrainSVM() {
  
  if(kernelid === 1) {
    trainstats= svm.train(data, labels, { kernel: svmjs.makeRbfKernel(rbfKernelSigma) , C: svmC});
  }
  if(kernelid === 0) {
    trainstats= svm.train(data, labels, { kernel: svmjs.linearKernel , C: svmC});
    wb= svm.getWeights();
  }
  if(kernelid === 2) {
    trainstats= svm.train(data, labels, { kernel: svmjs.polynomialKernel(polyC, polyD) , C: svmC});
  }
  if(kernelid === 3) {
    trainstats= svm.train(data, labels, { kernel: svmjs.sigmoidKernel(0, 3) , C: svmC});
  }
  
  dirty= true; // to redraw screen
}  
  
function update(){
}

function draw(){
    if(!dirty) return;
    
    ctx.clearRect(0,0,WIDTH,HEIGHT);
    
    // draw decisions in the grid
    var density= 3.0;
    for(var x=0.0; x<=WIDTH; x+= density) {
      for(var y=0.0; y<=HEIGHT; y+= density) {
        var dec= svm.marginOne([(x-WIDTH/2)/ss, (y-HEIGHT/2)/ss]);
        if(dec>0) ctx.fillStyle = 'rgb(150,250,150)';
        else ctx.fillStyle = 'rgb(250,150,150)';
        ctx.fillRect(x-density/2-1, y-density-1, density+2, density+2);
      }
    }
    
    // draw axes
    ctx.beginPath();
    ctx.strokeStyle = 'rgb(50,50,50)';
    ctx.lineWidth = 1;
    ctx.moveTo(0, HEIGHT/2);
    ctx.lineTo(WIDTH, HEIGHT/2);
    ctx.moveTo(WIDTH/2, 0);
    ctx.lineTo(WIDTH/2, HEIGHT);
    ctx.stroke();

    // draw datapoints. Draw support vectors larger
    ctx.strokeStyle = 'rgb(0,0,0)';
    for(var i=0;i<N;i++) {
      
      if(labels[i]==1) ctx.fillStyle = 'rgb(100,200,100)';
      else ctx.fillStyle = 'rgb(200,100,100)';
      
      if(svm.alpha[i]>1e-2) ctx.lineWidth = 3; // distinguish support vectors
      else ctx.lineWidth = 1;
      //ctx.lineWidth = 1;
      
      //drawCircle(data[i][0]*ss+WIDTH/2, data[i][1]*ss+HEIGHT/2, Math.floor(3+svm.alpha[i]*5.0/svmC));
      drawCircle(data[i][0]*ss+WIDTH/2, data[i][1]*ss+HEIGHT/2, 5);
    }
    
    // if linear kernel, draw decision boundary and margin lines
    if(kernelid == 0) {
    
      var xs= [-5, 5];
      var ys= [0, 0];
      ys[0]= (-wb.b - wb.w[0]*xs[0])/wb.w[1];
      ys[1]= (-wb.b - wb.w[0]*xs[1])/wb.w[1];
      ctx.fillStyle = 'rgb(0,0,0)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      // wx+b=0 line
      ctx.moveTo(xs[0]*ss+WIDTH/2, ys[0]*ss+HEIGHT/2);
      ctx.lineTo(xs[1]*ss+WIDTH/2, ys[1]*ss+HEIGHT/2);
      // wx+b=1 line
      ctx.moveTo(xs[0]*ss+WIDTH/2, (ys[0]-1.0/wb.w[1])*ss+HEIGHT/2);
      ctx.lineTo(xs[1]*ss+WIDTH/2, (ys[1]-1.0/wb.w[1])*ss+HEIGHT/2);
      // wx+b=-1 line
      ctx.moveTo(xs[0]*ss+WIDTH/2, (ys[0]+1.0/wb.w[1])*ss+HEIGHT/2);
      ctx.lineTo(xs[1]*ss+WIDTH/2, (ys[1]+1.0/wb.w[1])*ss+HEIGHT/2);
      ctx.stroke();
      
      // draw margin lines for support vectors. The sum of the lengths of these
      // lines, scaled by C is essentially the total hinge loss.
      for(var i=0;i<N;i++) {
        if(svm.alpha[i]<1e-2) continue;
        if(labels[i]==1) {
          ys[0]= (1 -wb.b - wb.w[0]*xs[0])/wb.w[1];
          ys[1]= (1 -wb.b - wb.w[0]*xs[1])/wb.w[1];
        } else {
          ys[0]= (-1 -wb.b - wb.w[0]*xs[0])/wb.w[1];
          ys[1]= (-1 -wb.b - wb.w[0]*xs[1])/wb.w[1];
        }
        var u= (data[i][0]-xs[0])*(xs[1]-xs[0])+(data[i][1]-ys[0])*(ys[1]-ys[0]);
        u = u/((xs[0]-xs[1])*(xs[0]-xs[1])+(ys[0]-ys[1])*(ys[0]-ys[1]));
        var xi= xs[0]+u*(xs[1]-xs[0]);
        var yi= ys[0]+u*(ys[1]-ys[0]);
        ctx.moveTo(data[i][0]*ss+WIDTH/2, data[i][1]*ss+HEIGHT/2);
        ctx.lineTo(xi*ss+WIDTH/2, yi*ss+HEIGHT/2);
      }
      ctx.stroke();
    }
    
    ctx.fillStyle= 'rgb(0,0,0)';
    ctx.fillText("Converged in " + trainstats.iters + " iterations.", 10, HEIGHT-30); 
    var numsupp=0;
    for(var i=0;i<N;i++) { if(svm.alpha[i] > 1e-5) numsupp++; }
    ctx.fillText("Number of support vectors: " + numsupp + " / " + N, 10, HEIGHT-50); 
    
    if(kernelid === 1) ctx.fillText("Using Rbf kernel with sigma = " + rbfKernelSigma.toPrecision(2), 10, HEIGHT - 70);
    if(kernelid === 0) ctx.fillText("Using Linear kernel", 10, HEIGHT - 70);
    if(kernelid === 2) ctx.fillText("Using Polynomial kernel", 10, HEIGHT - 70);
    if(kernelid === 3) ctx.fillText("Using Exponential kernel", 10, HEIGHT - 70);
    
    ctx.fillText("C = " + svmC.toPrecision(2), 10, HEIGHT - 90);
}

function mouseMousedown(x, y) {
    mx = (x-WIDTH/2)/ss;
    my = (y-HEIGHT/2)/ss;
    
    selecting:
    for(var i=0;i<N;i++) {
        if (Math.pow((mx-data[i][0]),2)+Math.pow((my-data[i][1]),2)<.03) {
            selected = i;
            break selecting;
        }
        selected = -1;
    }
}

function mouseMouseup(x, y, shiftPressed) {
    if (selected == -1) {
        // add datapoint at location of click
        data[N] = [(x-WIDTH/2)/ss, (y-HEIGHT/2)/ss];
        labels[N] = shiftPressed ? 1 : -1;
        N += 1;
        // retrain the svm
        retrainSVM();
    }
    else {
        selected = -1
        retrainSVM();
    }
}

function mouseMove(x, y, button){
    if (button==1 || selected != -1) {
        mx = (x-WIDTH/2)/ss;
        my = (y-HEIGHT/2)/ss;
        data[selected] = [mx,my];
        retrainSVM();
    }
}

function keyUp(key){

  if(key==82) { // 'r'
  
    // reset to original data and retrain
    data= data.splice(0, 2);
    labels= labels.splice(0, 2);
    N= 2;
    retrainSVM();
  }
  if(key==75) { // 'k'
    
    // toggle between kernels: rbf or linear
    kernelid= (kernelid + 1) % 4; // toggle 1 and 0
    retrainSVM();
  }
}
function keyDown(key){
}


// UI stuff
function refreshC(event, ui) {
  var logC = ui.value;
  svmC= Math.pow(10, logC);
  $("#creport").text("C = " + svmC.toPrecision(2));
  retrainSVM();
}

function refreshSig(event, ui) {
  var logSig = ui.value;
  rbfKernelSigma= Math.pow(10, logSig);
  $("#sigreport").text("RBF Kernel sigma = " + rbfKernelSigma.toPrecision(2));
  if(kernelid==1) {
    retrainSVM();	
  }
}

function refreshPolyD(event, ui) {
  polyD = ui.value;
  $("#polyDreport").text("Polynomial Kernel Dimension = " + polyD.toPrecision(2));
  if(kernelid==2) {
    retrainSVM();
  }
}

function refreshPolyC(event, ui) {
  polyC = ui.value;
  $("#polyCreport").text("Polynomial Kernel C = " + polyC.toPrecision(2));
  if(kernelid==2) {
    retrainSVM();
  }
}

$(function() {
        // for C parameter
		$("#slider1").slider({
			orientation: "horizontal",
			slide: refreshC,
			max: 5.0,
			min: -2.0,
			step: 0.1,
			value: 0.0
		});
		
		// for rbf kernel sigma
		$("#slider2").slider({
			orientation: "horizontal",
			slide: refreshSig,
			max: 2.0,
			min: -2.0,
			step: 0.1,
			value: 0.0
		});
		
		// for rbf kernel sigma
		$("#slider3").slider({
			orientation: "horizontal",
			slide: refreshPolyC,
			max: 2.0,
			min: -2.0,
			step: 0.1,
			value: 0.0
		});
		
		// for rbf kernel sigma
		$("#slider4").slider({
			orientation: "horizontal",
			slide: refreshPolyD,
			max: 5,
			min: 1,
			step: 1,
			value: 2
		});
});

function pauseSVM() {
    NPGPause(30, "btnReg");
}
