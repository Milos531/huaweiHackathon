import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { ModelArtsStroke } from '../model/modelartsstroke';
import { ModelArtsDiabetes } from '../model/modelartsdiabetes';
import { ModelartsService } from '../modelarts.service';

@Component({
  selector: 'app-form',
  templateUrl: './form.component.html',
  styleUrls: ['./form.component.css']
})

export class FormComponent {
  constructor (private router: Router, private modelarts: ModelartsService) {}
  age : number=0;
  sex : string;
  education : number=1;
  income : number=0;
  marrige : number;
  work : string;
  residence : string;
  highBP : number=0;
  highChol : number=0;
  cholTest : number=0;
  bmi : number=0;
  glucose : number=0;
  smoker : string;
  stroke : number = 0;
  heartDisease : number=0;
  pysichalActiv : number=0;
  fruit : number=0;
  vegetable : number=0;
  drink : number=0;
  healthcare : number=0;
  noMoney : number=0;
  genHealth : number=0;
  mentalHealth : number=0;
  pysichalHealth : number=0;
  stairs : number=0;
  
  
  submit(){
    var diabetes: ModelArtsDiabetes = new ModelArtsDiabetes()
    var stroke: ModelArtsStroke = new ModelArtsStroke()
    diabetes.highBP = this.highBP
    diabetes.highChol = this.highChol
    diabetes.sex = this.sex == "Male" ? 1 : 0
    diabetes.age = this.age
    diabetes.cholCheck = this.cholTest
    diabetes.bmi = this.bmi
    diabetes.smoker = this.smoker == "yes" ? 1 : 0
    diabetes.stroke = this.stroke
    diabetes.hearthDisease = this.heartDisease
    diabetes.physichalAchivity = this.pysichalActiv
    diabetes.fruit = this.fruit
    diabetes.veggies = this.vegetable
    diabetes.hvyAlcoholConsump = this.drink
    diabetes.anyHealthcare = this.healthcare
    diabetes.noDocbcCost = this.noMoney
    diabetes.genHlth = this.genHealth
    diabetes.mentalHlth = this.mentalHealth
    diabetes.physHlth = this.pysichalHealth
    diabetes.diffWalk = this.stairs
    diabetes.education = this.education
    diabetes.income = this.income

    stroke.gender = this.sex
    stroke.age = this.age
    stroke.hearth_disease = this.heartDisease
    stroke.ever_married = this.marrige==1 ? "Yes" : "No"
    stroke.work_type = this.work
    stroke.residence = this.residence
    stroke.bmi = this.bmi
    stroke.smoking_status = this.smoker
    stroke.hypertension = this.highBP;
    stroke.glucose = this.glucose;

    this.modelarts.submitDiabetes(diabetes).subscribe(()=>{
      alert("Diabetes!")
    })

    
    this.modelarts.submitStroke(stroke).subscribe((resp)=>{
      //alert("Stroke!")
      if(resp["result"] == 1)
      {
        alert("You are going to have a stroke, see a doctor immediately!");
      }
      else{
        alert("You have low chances of having a stroke, everything seems normal");
      }


    });

  }

}
