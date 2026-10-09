<?php
class train
{
    var $weights;
    var $biases;

    public function run()
    {

    }

    private function saveFile()
    {
        if (file_exists('../web/model.js')){
            @unlink('../web/model-bkp.js');
            rename('../web/model.js', '../web/model-bkp.js');
        }
        file_put_contents('../web/model.js', $this->getFileContent());
    }

    private function getFileContent()
    {
        $model = [
            'layers'  => [784, 64, 10],
            'weights' => $this->weights,   
            'biases'  => $this->biases,
        ];        
        return 'const MODEL = '.json_encode($model).';';
    }
}

(new train)->run();