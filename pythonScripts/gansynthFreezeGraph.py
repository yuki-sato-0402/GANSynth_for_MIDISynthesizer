import os
import sys
import argparse
import tensorflow.compat.v1 as tf

# Setting the import path for internal libraries
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from lib import model as lib_model
from lib import flags as lib_flags
from tensorflow.python.framework import graph_util

def freeze_gansynth(ckpt_dir, output_pb_path):
    tf.disable_v2_behavior()
    
    print(f"Loading model from: {ckpt_dir}")
    
    # Setting `eval_batch_size` to 1 fixes the shape of the placeholder to [1, ...].
    flags = lib_flags.Flags({
        'batch_size_schedule': [1],
        'eval_batch_size': 1,
        'train_root_dir': ckpt_dir,
        'tfds_data_dir': 'gs://tfds-data/datasets'  # Additional: Flags referenced when loading the model
    })
    
    # Load checkpoint
    # Model.load_from_path in lib/model.py automatically searches for the latest stage_xxxx folder.
    try:
        model = lib_model.Model.load_from_path(ckpt_dir, flags)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Verifying Input/Output Tensor Names
    # model.labels_ph: Pitch (int32, [1])
    # model.noises_ph: Latent variable (float32, [1, 256])
    # model.fake_data_ph: [1, 128, 1024, 2]
    
    input_node_names = [model.labels_ph.op.name, model.noises_ph.op.name]
    output_node_names = [model.fake_data_ph.op.name]
    
    #Input nodes: ['Placeholder', 'Placeholder_1']
    #Output nodes: ['Generator_1/strided_slice_9']

    print("Model Info for C++ / ONNX ---")
    print(f"Input nodes: {input_node_names}")
    print(f"Output nodes: {output_node_names}")
    print("---------------------------------")

    # 3. Freezing the graph (converting variables to constants)
    input_graph_def = model.sess.graph.as_graph_def()
    frozen_graph_def = graph_util.convert_variables_to_constants(
        model.sess,
        input_graph_def,
        output_node_names
    )
    
    # 4. Save as .pb file
    with tf.gfile.GFile(output_pb_path, "wb") as f:
        f.write(frozen_graph_def.SerializeToString())
    
    print(f"Success! Frozen graph saved to: {output_pb_path}")
    print(f"Next step: python -m tf2onnx.convert --input {output_pb_path} "
          f"--inputs {input_node_names[0]}:0,{input_node_names[1]}:0 "
          f"--outputs {output_node_names[0]}:0 --output gansynth.onnx")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #　Please prepare a pre-trained model.
    parser.add_argument("--ckpt_dir", type=str, default="./all_instruments", 
                        help="Path to the directory containing experiment.json and stage_xxxx folders")
    parser.add_argument("--output_pb", type=str, default="./model/gansynth_frozen.pb",
                        help="Output filename for the frozen graph")
    args = parser.parse_args()
    
    freeze_gansynth(args.ckpt_dir, args.output_pb)


"""
python -m tf2onnx.convert --input ./model/gansynth_frozen.pb \
        --inputs "Placeholder:0,Placeholder_1:0" \
        --outputs "Generator_1/truediv:0" \
        --opset 16 \
        --output ./model/gansynth.onnx
"""