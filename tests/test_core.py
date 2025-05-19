def test_basic_run():
    import pandas as pd
    from automl import AutoML

    df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv')
    df.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']

    model = AutoML(df, target='Outcome')
    result = model.run()
    assert result is not None