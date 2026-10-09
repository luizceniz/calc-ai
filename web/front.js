var operation = '+';

$(document).ready(function(){
    
    // operation btns
    $('.operation button').each(function(){
        $(this).on('click', function(){
            $('.operation button').removeClass('sel');
            $(this).addClass('sel');
            operation = $(this).text();
        })
    });

    // canvas
    $('canvas').each(function(){
        const ctx = this.getContext("2d");
        ctx.lineWidth = 3;
        ctx.strokeStyle = "black";
        ctx.lineCap = "round";
        $(this).data("drawing", false);

        // canvas events
        $(this).on('mousedown', function(e){
            $(this).data("drawing", true);
            draw(this, e);
        });
        $(this).on('mouseup', function(){
            stopDraw(this);
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
    $(canvaObj).data("drawing", false);
    const ctx = canvaObj.getContext("2d");    
    ctx.beginPath();
}

function draw(canvaObj, e)
{
    if (!$(canvaObj).data("drawing")) return;

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
    const canvas = document.getElementById("digit"+canvaIdx);
    const ctx = canvas.getContext("2d");    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}