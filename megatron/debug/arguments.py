def add_debug_args(parser):
    group = parser.add_argument_group(title='debug')

    group.add_argument('--log-tensor-name-pattern', type=str, default="",
                       help='The module name pattern by which log tensor is applied')
    group.add_argument('--log-tensor-interval', type=int, default=1,
                       help='Log tensor interval')
    return parser


