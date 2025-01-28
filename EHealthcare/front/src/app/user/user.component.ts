import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { User } from '../model/user';

@Component({
  selector: 'app-user',
  templateUrl: './user.component.html',
  styleUrls: ['./user.component.css']
})
export class UserComponent implements OnInit{

  constructor(private router: Router){}

  ngOnInit(): void {
    this.user = JSON.parse(localStorage.getItem('user'))
    if (!this.user || this.user.type != "patient"){
      this.router.navigate([''])
    }
  }

  user: User = null
  

  logout(){
    localStorage.clear()
    this.router.navigate([''])
  }
}