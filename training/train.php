<?php
class train
{
    var $weights;
    var $biases;
    var $layers = [784, 64, 10];   // input pixels, hidden neurons, output digits

    public function run()
    {
        $this->initWeights();

        $limit = 10000;    // samples per epoch
        $rate  = 0.1;      // learning rate
        $start = time();

        $csv = fopen('train.csv', 'r');

        $n = 0;
        $errorSum = 0;
        $hits = 0;

        while ($n < $limit && ($fields = fgetcsv($csv))) {
            $label  = (int) array_shift($fields);                // first value: the correct digit
            $pixels = array_map(fn($p) => $p / 255, $fields);    // 784 pixels scaled to 0..1

            $activations = $this->forward($pixels);

            // was it right? (checked BEFORE learning from this sample)
            $chances = $activations[2];
            if (array_search(max($chances), $chances) == $label) $hits++;

            $errorSum += $this->backprop($activations, $label, $rate);
            $n++;

            // show a line every 1000 samples, then reset the counters
            if ($n % 1000 == 0) {
                $this->showProgress($n, $limit, $errorSum / 1000, $hits / 10, $start);
                $errorSum = 0;
                $hits = 0;
            }
        }
        fclose($csv);

        $this->saveFile();

        echo "New model.js has been saved!!\n\n";
    }

    private function showProgress($n, $limit, $error, $accuracy, $start)
    {
        // 20 blocks = 100%
        $filled = (int) ($accuracy / 5);
        $bar = str_repeat('█', $filled) . str_repeat('░', 20 - $filled);
        printf("%5d/%d | error %.3f | hit %5.1f%% %s | %ds\n",
            $n, $limit, $error, $accuracy, $bar, time() - $start);
    }

    private function initWeights()
    {
        for ($l = 0; $l < count($this->layers) - 1; $l++) {
            $nIn  = $this->layers[$l];        // neurons sending
            $nOut = $this->layers[$l + 1];    // neurons receiving

            // start small: weights within ±1/sqrt(inputs) keep each neuron's sum
            // around 1, so the sigmoid starts in its steep (learning) region
            $limit = 1 / sqrt($nIn);

            for ($j = 0; $j < $nOut; $j++) {
                for ($i = 0; $i < $nIn; $i++) {
                    $random = mt_rand() / mt_getrandmax();                    // 0..1
                    $this->weights[$l][$j][$i] = ($random * 2 - 1) * $limit;  // -limit..+limit
                }
                $this->biases[$l][$j] = 0;
            }
        }
    }

    private function forward($input)
    {
        $activations = [$input];          // layer 0 = the 784 pixels
        $values = $input;

        foreach ($this->weights as $l => $layerWeights) {
            $isLast = $l == count($this->weights) - 1;

            // for each neuron: weight × value of each incoming connection, sum it all + bias
            $z = [];
            foreach ($layerWeights as $j => $neuronWeights) {
                $sum = $this->biases[$l][$j];
                foreach ($neuronWeights as $i => $w) {
                    $sum += $w * $values[$i];
                }
                $z[$j] = $sum;
            }

            // activation: this layer's output becomes the next layer's input
            $values = $isLast ? $this->softmax($z) : array_map(fn($v) => $this->sigmoid($v), $z);
            $activations[] = $values;
        }

        return $activations;   // [pixels, hidden outputs, final chances]
    }

    private function sigmoid($v)
    {
        return 1 / (1 + exp(-$v));
    }

    private function softmax($z)
    {
        $max = max($z);                                   // subtracting the max avoids overflow
        $exps = array_map(fn($v) => exp($v - $max), $z);
        $total = array_sum($exps);
        return array_map(fn($v) => $v / $total, $exps);   // chances that add up to 1
    }

    private function backprop($activations, $label, $rate)
    {
        $input  = $activations[0];   // 784 pixels
        $hidden = $activations[1];   // 64 hidden outputs
        $output = $activations[2];   // 10 chances

        // 1) target: all zeros, with 1 at the correct digit's position
        $target = array_fill(0, 10, 0);
        $target[$label] = 1;

        // 2) blame of each output: output − target
        $deltaOut = [];
        foreach ($output as $k => $o) {
            $deltaOut[$k] = $o - $target[$k];
        }

        // 3) blame of each hidden neuron: (Σ weight to each output × that output's blame) × a(1−a)
        //    must run BEFORE step 4a, which changes those weights
        $deltaHidden = [];
        foreach ($hidden as $j => $a) {
            $sum = 0;
            foreach ($deltaOut as $k => $d) {
                $sum += $this->weights[1][$k][$j] * $d;
            }
            $deltaHidden[$j] = $sum * $a * (1 - $a);
        }

        // 4a) adjust output weights and biases
        foreach ($deltaOut as $k => $d) {
            foreach ($hidden as $j => $a) {
                $this->weights[1][$k][$j] -= $rate * $d * $a;
            }
            $this->biases[1][$k] -= $rate * $d;
        }

        // 4b) adjust hidden weights and biases
        foreach ($deltaHidden as $j => $d) {
            foreach ($input as $i => $x) {
                if ($x == 0) continue;     // empty pixel: no blame, the adjustment would be zero
                $this->weights[0][$j][$i] -= $rate * $d * $x;
            }
            $this->biases[0][$j] -= $rate * $d;
        }

        // error (cross-entropy): small when the network gave a high chance to the correct digit
        return -log(max($output[$label], 1e-12));
    }

    private function saveFile()
    {
        // keep the previous model as a backup
        if (file_exists('../web/model.js')) {
            @unlink('../web/model-bkp.js');
            rename('../web/model.js', '../web/model-bkp.js');
        }
        file_put_contents('../web/model.js', $this->getFileContent());
    }

    private function getFileContent()
    {
        $model = [
            'layers'  => $this->layers,
            'weights' => $this->weights,
            'biases'  => $this->biases,
        ];
        return 'const MODEL = ' . json_encode($model) . ';';
    }
}

(new train)->run();