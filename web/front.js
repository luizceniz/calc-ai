var operation = '+';

$(document).ready(function(){
    
    // operation btns
    $('.operation button').each(function(){
        $(this).on('click', function(){
            $('.operation button').removeClass('sel');
            $(this).addClass('sel');
            operation = $(this).text();
            tryAutoCalc();
        })
    });

    // drawings
    $('canvas').each(function(){
        const ctx = this.getContext("2d");
        ctx.lineWidth = 30;
        ctx.strokeStyle = "black";
        ctx.lineCap = "round";
        $(this).attr("drawing", 0);

        // canvas events
        $(this).on('mousedown', function(e){
            $(this).attr("drawing", 1);
            objCanvas.clear(this.id[5]);
            draw(this, e);
        });
        $(this).on('mouseup', function(){
            stopDraw(this);
            tryAutoCalc();
        });
        $(this).on('mouseleave', function(){
            stopDraw(this);
        });
        $(this).on('mousemove', function(e){
            draw(this, e);
        });
    });

})

function stopDraw(canvaObj)
{
    $(canvaObj).attr("drawing", 0);
    const ctx = canvaObj.getContext("2d");    
    ctx.beginPath();
}

function tryAutoCalc()
{
    try{
        const digit1 = objCanvas.getDigit(1);
        const digit2 = objCanvas.getDigit(2);
        calc(digit1, digit2);
    } catch(e){
        // ignore
    }
}

function draw(canvaObj, e)
{
    if ($(canvaObj).attr("drawing") != 1){
        return;
    }

    $('.res').text(''); //  cleaning result

    // mouse position
    const rect = canvaObj.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const ctx = canvaObj.getContext("2d");    

    ctx.lineTo(x, y);
    ctx.stroke();  
    ctx.beginPath();
    ctx.moveTo(x, y);    
}

function clear(canvaIdx)
{
    objCanvas.clear(canvaIdx);
}

function calc(d1, d2)
{
    try{
        const digit1 = d1 !== undefined ? d1 : objCanvas.getDigit(1);
        const digit2 = d2 !== undefined ? d2 : objCanvas.getDigit(2);       
        
        const calc = digit1+operation+digit2;

        console.log(calc);

        const res = eval(calc);

        $('.res').text(res);
    } catch(e){
        console.error(e.message);
        $('.res').text('');
    }
    return false;
}


const objBrain = {
    predict(values){
        MODEL.weights.forEach((layerWeights, l) => {
            const isLast = (l === MODEL.weights.length - 1);

            // for each cell, weight  × value of each line, sum it all + bias
            const z = layerWeights.map((neuronWeights, j) => {
                let sum = MODEL.biases[l][j];
                neuronWeights.forEach((w, i) => sum += w * values[i]);
                return sum;
            });

            // activation. The layer output becomes input for the next layer
            values = isLast ? this.softmax(z) : z.map(v => 1 / (1 + Math.exp(-v)));
        });

        return values;                    
    },

    softmax(z) {
        const max = Math.max(...z);
        const exps = z.map(v => Math.exp(v - max));
        const total = exps.reduce((a, b) => a + b, 0);
        return exps.map(v => v / total);
    }
};    

const objCanvas = {
    clear(idx){
        const canvas = document.getElementById("digit"+idx);
        const ctx = canvas.getContext("2d");    
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    },

    getDigit(idx){
        const canvas = document.getElementById('digit'+idx);
        const rect = this.fitRect(canvas);
        if (!rect){
            throw new Error(`The digit ${idx} is empty`);
        }   
        const drawData = this.getDrawData(idx, rect);
        const chances = objBrain.predict(drawData);

        console.log(idx, chances);

        // finding the bigger chance
        return chances.indexOf(Math.max(...chances));   
    },

    getDrawData(idx, rect){
        const newCanva = this.buildSmallCanva(idx, rect);

        const ctx = newCanva.getContext('2d');
        const pixels = ctx.getImageData(0, 0, newCanva.width, newCanva.height).data;
        let data = [];

        for (let y = 0; y < newCanva.height; y++) {
            for (let x = 0; x < newCanva.width; x++) {
                const i = (y * newCanva.width + x) * 4;          
                data.push(pixels[i + 3] / 255); 
            }
        }

        return this.centerByMass(data);
    },

    centerByMass(data) {
        // mass center, average by paint
        let total = 0, sumX = 0, sumY = 0;
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const p = data[y * 28 + x];
                total += p;
                sumX  += x * p;
                sumY  += y * p;
            }
        }
        const cx = sumX / total;
        const cy = sumY / total;

        // shift the mass center
        const shiftX = Math.round(13.5 - cx);
        const shiftY = Math.round(13.5 - cy);

        // copying each pixel
        const moved = new Array(784).fill(0);
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const nx = x + shiftX, ny = y + shiftY;
                if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28) {   
                    moved[ny * 28 + nx] = data[y * 28 + x];
                }
            }
        }

        return moved;
    },    

    buildSmallCanva(idx, rect){
        // detect bigger measure
        var newW, newH;
        if (rect.width > rect.height){
            newW = 20;
            newH = Math.round( (20 / rect.width) * rect.height);
        } else {
            newH = 20;
            newW = Math.round( (20 / rect.height) * rect.width);
        }

        const canvas = document.getElementById('digit'+idx);

        var newCanva = document.createElement('canvas');
        newCanva.width=28;
        newCanva.height=28;
        const ctx = newCanva.getContext('2d');
        ctx.imageSmoothingEnabled  = true;
        ctx.imageSmoothingQuality = 'high';

        // copying, resampling and centering at the same time
        ctx.drawImage(canvas, rect.x, rect.y, rect.width, rect.height, 
                    Math.round((28 - newW) / 2), Math.round((28 - newH) / 2), newW, newH);

        return newCanva;
    },

    fitRect(canvas) {
        const ctx = canvas.getContext('2d');
        const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        let minX = canvas.width, minY = canvas.height, maxX = -1, maxY = -1;

        for (let y = 0; y < canvas.height; y++) {
            for (let x = 0; x < canvas.width; x++) {
                const i = (y * canvas.width + x) * 4;          
                const hasPaint = pixels[i + 3] > 0 && pixels[i] < 128; 
                if (hasPaint) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }

        // check empty canvas
        if (maxX < 0) return null;                          
        return { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1 };
    }
   
}
