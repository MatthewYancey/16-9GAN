# Project Notes

## Backlog
* Use image hash

## Results to show
* Results over time
* Example on the 16:9
* Example on the 4:3
    * fixed background vs action sceen
* Look at other animated shows (Simpsons etc)
* Look at the wizard of oz

## Links and Literature
Berkey's project of filling in missing information from images. http://people.eecs.berkeley.edu/~pathak/context_encoder/#extraResults
Medium article reviewing different types of inpainting https://towardsdatascience.com/10-papers-you-must-read-for-deep-image-inpainting-2e41c589ced0
Globally and Locally Consistent Inpainting http://iizuka.cs.tsukuba.ac.jp/projects/completion/data/completion_sig2017.pdf
- Some tests were run on 100k images for 500 epochs
- How is masking done? The input of the completion network is an RGB image with a binary channel that indicates the image completion mask (1 for a pixel to be completed)
- They did do some networks with just weighted MSE (what is the weighting?)

## Notes
Looked into OpenAI's Image GPT (https://openai.com/blog/image-gpt/). It may take too long to train even if it's just transfer learning on this model.
Code is on github (https://github.com/openai/image-gpt). Was created with TensorFlow.
30 is the largest batch size
