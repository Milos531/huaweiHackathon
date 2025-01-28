import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DoctorComponent } from './doctor/doctor.component';
import { FooterComponent } from './footer/footer.component';
import { FormComponent } from './form/form.component';
import { HeaderComponent } from './header/header.component';
import { HomeComponent } from './home/home.component';
import { LoginComponent } from './login/login.component';
import { RegisterComponent } from './register/register.component';
import { UserComponent } from './user/user.component';

const routes: Routes = [
  {path: "form", component: FormComponent},
  {path: "", component: HomeComponent, children: [
      {path: "login", component: LoginComponent},
      {path: "register", component: RegisterComponent},
  ]},
  {path: "patient", component: UserComponent},
  {path: "doctor", component: DoctorComponent }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule { }
