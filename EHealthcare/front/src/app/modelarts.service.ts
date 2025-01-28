import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { ModelArtsDiabetes } from './model/modelartsdiabetes';
import { ModelArtsStroke } from './model/modelartsstroke';

@Injectable({
  providedIn: 'root'
})
export class ModelartsService {

  constructor(private http: HttpClient) { }

  urlDiabetes = 'http://'
  urlStroke = 'http://localhost:8080/stroke'

  submitDiabetes(diabetes: ModelArtsDiabetes){
    return this.http.post(`${this.urlDiabetes}`, diabetes)
  }



  submitStroke(stroke: ModelArtsStroke ){
    const data = {
      gender: stroke.gender,
      age: stroke.age,
      hypertension: stroke.hypertension,
      heart_disease: stroke.hearth_disease,
      ever_married: stroke.ever_married,
      work_type: stroke.work_type,
      Residence_type: stroke.residence,
      avg_glucose_level: stroke.glucose,
      bmi: stroke.bmi,
      smoking_status: stroke.smoking_status
    }
    return this.http.post(`${this.urlStroke}`,data);
  }






}
