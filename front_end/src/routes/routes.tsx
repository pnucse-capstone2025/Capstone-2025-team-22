import { createBrowserRouter, RouterProvider} from 'react-router-dom'
import { RouterPath }                     from '@/routes/path'
import { HomePage }                       from '@/pages/Home/Home'
import { DetailPage }                     from '@/pages/Detail/Detail'

const router = createBrowserRouter([
  {
    path: RouterPath.HOME,  
    element: <HomePage />,
  },
  {
    path: RouterPath.DETAIL,     
    element: <DetailPage />,
  },
])

export const Routes = () => (
  <RouterProvider router={router} />
)
