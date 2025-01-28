import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { User } from '../model/user';

@Component({
  selector: 'app-doctor',
  templateUrl: './doctor.component.html',
  styleUrls: ['./doctor.component.css']
})
export class DoctorComponent {
  constructor(private router: Router){}

  ngOnInit(): void {
    this.user = JSON.parse(localStorage.getItem('user'))
    if (!this.user || this.user.type != "doctor"){
      this.router.navigate([''])
    }
  }

  user: User = null
  

  logout(){
    localStorage.clear()
    this.router.navigate([''])
  }
}
