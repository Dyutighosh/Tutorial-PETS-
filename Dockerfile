# This pulls pytorch image with Python 3.7
FROM pytorch/pytorch

RUN  pip install omegaconf
RUN  pip install scikit-learn
RUN  pip install tensorboard
RUN  pip install torchvision
RUN  pip install onnx
RUN  pip install matplotlib
RUN  pip install crypten --no-deps
RUN  pip install jupyter


COPY Intro.ipynb Intro.ipynb
COPY Training_a_Model_on_Encrypted_Data.ipynb Training_a_Model_on_Encrypted_Data.ipynb
COPY Data_Across_Multiple_Parties.ipynb Data_Across_Multiple_Parties.ipynb

# start jupyter notebook
# Tini operates as a process subreaper for jupyter to prevent kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--port=8891", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
