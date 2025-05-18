Python -m venv env
.\env\Scripts\activate #windows
source env/bin/activate # linux/macos

deactivate

pip install -r requirements.txt

python -m ipykernel install --user --name=env --display-name "Python env"


mlflow ui
python Membangun_model/modelling.py
python Membangun_model/modelling_tuning.py



