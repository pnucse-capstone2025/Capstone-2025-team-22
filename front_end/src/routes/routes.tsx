import { createBrowserRouter, RouterProvider} from 'react-router-dom'
import { RouterPath }                     from './path'
import { MainLayout }                     from '../layouts/MainLayout'
import { HomePage }                       from '../pages/Home/Home'
import { DetailPage }                     from '../pages/Detail/Detail'

const router = createBrowserRouter([
  {
    path: RouterPath.HOME,  
    element: <MainLayout />,    
    children: [
      {
        index: true,           
        element: <HomePage />,
      },
      {
        path: 'detail/:id',     
        element: <DetailPage />,
      },
    ],
  },
])

export const Routes = () => (
  <RouterProvider router={router} />
)
